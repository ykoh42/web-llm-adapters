import type { ChatCompletionChunk, ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import { ModelError } from './types.js'
import { stringifyJSON } from './json.js'
import { stopReason, type StreamChunk, type ChunkOptions } from './stream/chunks.js'

interface NativeCall {
  id?: string
  name: string
  input: string
}

/** WebLLM streams Hermes JSON first, then exposes the same output as parsed calls. */
export async function* nativeChunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  hasTools: boolean,
  { onChunk, signal }: ChunkOptions<ChatCompletionChunk> = {},
): AsyncGenerator<StreamChunk> {
  const calls = new Map<number, NativeCall>()
  let pending = ''
  let mode: 'start' | 'json' | 'text' = hasTools ? 'start' : 'text'

  function decode(chunk: ChatCompletionChunk): StreamChunk {
    const choice = chunk.choices[0]
    if (chunk.choices.length > 1 || (choice && choice.index !== 0))
      throw new ModelError('WebLLM adapters need one response per request.')

    const deltas = choice?.delta.tool_calls ?? []
    for (const delta of deltas) {
      const call: NativeCall = calls.get(delta.index) ?? { name: '', input: '' }
      if (call.id && delta.id && call.id !== delta.id)
        throw new ModelError('WebLLM changed a tool call ID while streaming.')
      if (delta.id) call.id = delta.id
      call.name += delta.function?.name ?? ''
      call.input += delta.function?.arguments ?? ''
      calls.set(delta.index, call)
    }

    const stop = stopReason(choice?.finish_reason)
    let text = choice?.delta.content ?? ''
    if (mode !== 'text') {
      pending += text
      if (mode === 'start') {
        const first = pending.trimStart()[0]
        if (first) mode = first === '[' ? 'json' : 'text'
      }
      text = ''
      if (mode === 'text' || stop) {
        const ordered = [...calls.entries()].sort(([left], [right]) => left - right)
        const duplicate =
          stop !== 'maxTokens' &&
          stop !== 'cancelled' &&
          matchesToolText(
            pending,
            ordered.map(([, call]) => call),
          )
        text = duplicate ? '' : pending
        pending = ''
      }
    }

    return {
      text,
      reasoning: '',
      calls: deltas.map((call) => ({
        index: call.index,
        id: call.id,
        name: call.function?.name,
        input: call.function?.arguments,
      })),
      stop,
      usage: chunk.usage,
    }
  }

  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      onChunk?.(chunk)
      if (signal?.aborted) return
      yield decode(chunk)
    }
  } catch (error) {
    failure = { error }
  }
  // Keep received bytes without inventing a finish reason or executing partial calls.
  if (signal?.aborted) return
  if (pending) yield { text: pending, reasoning: '', calls: [], stop: undefined, usage: undefined }
  if (failure && !signal?.aborted) throw failure.error
}

function matchesToolText(
  text: string,
  calls: readonly NativeCall[],
  allowExtraFields = false,
): boolean {
  if (!calls.length || text.trimStart()[0] !== '[') return false
  try {
    const output: unknown = JSON.parse(text)
    if (!Array.isArray(output) || output.length !== calls.length) return false
    return output.every((item: unknown, index) => {
      const call = calls[index]!
      if (!isRecord(item) || item.name !== call.name || !('arguments' in item)) return false
      const serialized = stringifyJSON(item.arguments)
      const input: unknown = JSON.parse(call.input)
      const same = serialized === stringifyJSON(input) || sameJSON(item.arguments, input)
      return same && (allowExtraFields || Object.keys(item).length === 2)
    })
  } catch (cause) {
    // WebLLM reserializes arguments, which can turn overflowing numbers into null.
    if (cause instanceof TypeError) throw cause
    return false
  }
}

function sameJSON(left: unknown, right: unknown): boolean {
  if (left === right) return true
  if (Array.isArray(left))
    return (
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value: unknown, index) => sameJSON(value, right[index]))
    )
  if (!isRecord(left) || !isRecord(right)) return false
  const keys = Object.keys(left)
  return (
    keys.length === Object.keys(right).length &&
    keys.every((key) => Object.hasOwn(right, key) && sameJSON(left[key], right[key]))
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
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
    if (matchesToolText(content, calls, true)) return message
    return {
      ...message,
      content:
        content +
        JSON.stringify(
          calls.map((call) => ({ name: call.name, arguments: JSON.parse(call.input) })),
        ),
    }
  })
}
