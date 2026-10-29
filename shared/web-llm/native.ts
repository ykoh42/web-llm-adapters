import type {
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from '@mlc-ai/web-llm'
import { ModelError, type ToolSpec } from './types.js'
import { sameJSON, stringifyJSON } from './json.js'
import { hasNegativeZero, toolJSON } from './tool-json.js'
import { callChunkCallback } from './callback.js'
import { throwStreamError } from './stream/failure.js'
import {
  singleChoice,
  snapshotChatChunk,
  stopReason,
  noToolCalls,
  type StreamChunk,
  type ChunkOptions,
  type ToolDelta,
} from './stream/chunks.js'

interface NativeCall {
  id?: string
  name: string
  input: string
}

/** WebLLM streams Hermes JSON first, then exposes the same output as parsed calls. */
export async function* nativeChunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  hasTools: boolean,
  { onChunk, onChunkMayMutate = true, signal }: ChunkOptions<ChatCompletionChunk> = {},
): AsyncGenerator<StreamChunk> {
  let calls: Map<number, NativeCall> | undefined
  let ids: Map<string, number> | undefined
  let pending = ''
  let mode: 'start' | 'json' | 'text' = hasTools ? 'start' : 'text'
  let arrayStart = false

  function recordToolDeltas(
    deltas: NonNullable<ChatCompletionChunk['choices'][number]['delta']['tool_calls']>,
  ): void {
    calls ??= new Map()
    ids ??= new Map()
    for (const delta of deltas) {
      const call: NativeCall = calls.get(delta.index) ?? { name: '', input: '' }
      if (call.id && delta.id && call.id !== delta.id)
        throw new ModelError('WebLLM changed a tool call ID while streaming.')
      if (delta.id) {
        const index = ids.get(delta.id)
        if (index !== undefined && index !== delta.index)
          throw new ModelError('WebLLM returned duplicate tool call IDs.')
        ids.set(delta.id, delta.index)
        call.id = delta.id
      }
      call.name += delta.function?.name ?? ''
      call.input += delta.function?.arguments ?? ''
      calls.set(delta.index, call)
    }
  }

  function isDuplicateText(stop: StreamChunk['stop'], outputCalls: readonly ToolDelta[]): boolean {
    if (!stop || stop === 'maxTokens' || stop === 'cancelled' || !calls?.size) return false
    const ordered = [...calls.entries()].sort(([left], [right]) => left - right)
    const output = parseToolText(pending)
    restoreNegativeZeros(pending, output, ordered, outputCalls)
    return matchesToolText(
      output,
      ordered.map(([, call]) => call),
    )
  }

  function readStart(text: string): string {
    const start = text.trimStart()
    if (!start) return text
    mode = start[0] === '[' ? 'json' : 'text'
    arrayStart = mode === 'json'
    return start.slice(1)
  }

  function readArrayStart(text: string): 'start' | 'json' | 'text' {
    const first = text.trimStart()[0]
    if (!first) return mode
    arrayStart = false
    // A native tool-call array can only contain objects.
    return first === '{' ? mode : 'text'
  }

  function decode(chunk: Pick<ChatCompletionChunk, 'choices' | 'usage'>): StreamChunk {
    const choice = singleChoice(chunk.choices)

    const deltas = choice?.delta.tool_calls
    if (deltas?.length) recordToolDeltas(deltas)

    const stop = stopReason(choice?.finish_reason)
    const outputCalls: readonly ToolDelta[] = deltas?.length
      ? deltas.map((call) => ({
          index: call.index,
          id: call.id,
          name: call.function?.name,
          input: call.function?.arguments,
        }))
      : noToolCalls
    let text = choice?.delta.content ?? ''
    if (mode !== 'text') {
      pending += text
      const arrayText = mode === 'start' ? readStart(text) : text
      if (arrayStart) mode = readArrayStart(arrayText)
      text = ''
      if (mode === 'text' || stop) {
        text = isDuplicateText(stop, outputCalls) ? '' : pending
        pending = ''
      }
    }

    return {
      text,
      reasoning: '',
      calls: outputCalls,
      stop,
      usage: chunk.usage,
    }
  }

  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      const parserChunk = onChunk && onChunkMayMutate ? snapshotChatChunk(chunk) : chunk
      const callbackResult = onChunk && callChunkCallback(onChunk, chunk, signal)
      if (callbackResult) await callbackResult
      if (signal?.aborted) return
      yield decode(parserChunk)
    }
  } catch (error) {
    failure = { error }
  }
  // Keep received bytes without inventing a finish reason or executing partial calls.
  if (signal?.aborted) return
  if (pending)
    yield { text: pending, reasoning: '', calls: noToolCalls, stop: undefined, usage: undefined }
  throwStreamError(failure, signal)
}

/** WebLLM reserializes native tool arguments and can change negative zero to zero. */
function restoreNegativeZeros(
  text: string,
  output: unknown,
  ordered: [number, NativeCall][],
  deltas: readonly ToolDelta[],
) {
  if (!text.includes('-')) return
  if (!Array.isArray(output) || output.length !== ordered.length) return
  if (!output.some((item: unknown) => isRecord(item) && hasNegativeZero(item.arguments))) return
  const values: { index: number; call: NativeCall; arguments: unknown; parsed: unknown }[] = []
  for (const [position, [index, call]] of ordered.entries()) {
    const item: unknown = output[position]
    if (!isRecord(item) || item.name !== call.name || !('arguments' in item)) return
    let parsed: unknown
    try {
      parsed = JSON.parse(call.input)
    } catch {
      return
    }
    if (!sameJSON(item.arguments, parsed)) return
    values.push({ index, call, arguments: item.arguments, parsed })
  }
  for (const value of values) {
    if (sameJSON(value.arguments, value.parsed, true)) continue
    const matches = deltas.filter((delta) => delta.index === value.index)
    const delta = matches[0]
    if (matches.length !== 1 || !delta?.input || delta.input !== value.call.input)
      throw new ModelError('WebLLM changed negative zero in a streamed tool call.')
    delta.input = toolJSON(value.arguments, text)
    value.call.input = delta.input
  }
}

function parseToolText(text: string): unknown {
  if (text.trimStart()[0] !== '[') return
  try {
    return JSON.parse(text)
  } catch {
    return
  }
}

function matchesToolText(output: unknown, calls: readonly NativeCall[], allowExtraFields = false) {
  if (!calls.length) return false
  try {
    if (!Array.isArray(output) || output.length !== calls.length) return false
    return output.every((item: unknown, index) => {
      const call = calls[index]!
      if (!isRecord(item) || item.name !== call.name || !('arguments' in item)) return false
      const serialized = stringifyJSON(item.arguments)
      let same = serialized === call.input
      if (!same) {
        const input: unknown = JSON.parse(call.input)
        same = serialized === stringifyJSON(input) || sameJSON(item.arguments, input, true)
      }
      return same && (allowExtraFields || Object.keys(item).length === 2)
    })
  } catch (cause) {
    // WebLLM reserializes arguments, which can turn overflowing numbers into null.
    if (cause instanceof TypeError) throw cause
    return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function nativeTools(tools: ToolSpec[]): ChatCompletionTool[] {
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: { ...(tool.inputSchema ?? { type: 'object', properties: {} }) },
    },
  }))
}

/** WebLLM reads assistant content but ignores tool_calls when rebuilding native history. */
export function nativeHistory(
  messages: ChatCompletionMessageParam[],
  modelId: string,
  hasTools = false,
): ChatCompletionMessageParam[] {
  if (!modelId.startsWith('Hermes-2-Pro-') && !modelId.startsWith('Hermes-3-')) return messages
  // WebLLM supplies Hermes tool instructions and rejects even an empty system message.
  const first = messages[0]
  if (hasTools && first?.role === 'system' && first.content === '') messages = messages.slice(1)
  return messages.map((message) => {
    if (message.role !== 'assistant' || !message.tool_calls?.length) return message
    const calls = message.tool_calls.map((call) => ({
      name: call.function.name,
      input: call.function.arguments,
    }))
    const content = message.content ?? ''
    // Keep optional native fields already present in history without repeating the calls.
    if (matchesToolText(parseToolText(content), calls, true)) return message
    return {
      ...message,
      content:
        content +
        stringifyJSON(
          calls.map((call) => ({ name: call.name, arguments: JSON.parse(call.input) })),
        ),
    }
  })
}
