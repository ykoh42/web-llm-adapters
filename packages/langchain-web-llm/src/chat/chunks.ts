import { checkToolCallIds, checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { parseJSON } from '../../../../shared/web-llm/json.js'
import type { ToolChoice } from '../../../../shared/web-llm/types.js'
import { AIMessageChunk, type StandardMessageStructure } from '@langchain/core/messages'
import { stampRetryable } from '@langchain/core/errors'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import {
  hasContentAfterStop,
  type StreamChunk,
  type ToolDelta,
} from '../../../../shared/web-llm/stream/chunks.js'
import { isQwen, isQwen35 } from '../../../../shared/web-llm/models.js'

interface ToolCall extends ToolDelta {
  readonly index: number
  name: string
  input: string
  pending: { delta: ToolDelta; name?: string }[]
  streamed: boolean
}

type Deferred = ToolCall | ChatGenerationChunk

const contentFields = [
  ['reasoning', 'reasoning'],
  ['text', 'text'],
] as const

/** Stream arguments without an ID until the complete call is validated. */
export async function* readChunks(
  chunks: AsyncIterable<StreamChunk>,
  modelId: string,
  signal?: AbortSignal,
  choice?: ToolChoice | 'none',
  toolNames?: readonly string[],
): AsyncGenerator<ChatGenerationChunk> {
  const calls = new Map<number, ToolCall>()
  let stop: StreamChunk['stop']
  let usage: StreamChunk['usage']
  let block: 'text' | 'reasoning' | undefined
  let index = -1
  let nextToolCallIndex = 0
  const streamTools = canStreamToolCalls(choice)
  const orderedTools = !streamTools && choice !== 'none' && (isQwen(modelId) || isQwen35(modelId))
  let deferred: Deferred[] | undefined
  let deferredIndex = 0

  function* emitDeferred(includeTools: boolean): Generator<ChatGenerationChunk> {
    while (deferred && deferredIndex < deferred.length) {
      const part = deferred[deferredIndex++]!
      if (part instanceof ChatGenerationChunk) yield part
      else if (includeTools) yield toolCallChunk(part)
      signal?.throwIfAborted()
    }
  }

  function recordToolDelta(delta: ToolDelta): ToolCall {
    let call = calls.get(delta.index)
    if (!call) {
      call = {
        index: delta.index,
        id: undefined,
        name: '',
        input: '',
        pending: [],
        streamed: false,
      }
      calls.set(delta.index, call)
      if (orderedTools) {
        deferred ??= []
        deferred.push(call)
      }
    }
    if (delta.id) {
      if (call.id && delta.id !== call.id) throw new Error('WebLLM changed a tool call ID.')
      call.id = delta.id
    }
    call.name += delta.name ?? ''
    call.input += delta.input ?? ''
    return call
  }

  function finishChunk(
    stop: NonNullable<StreamChunk['stop']>,
    usage: StreamChunk['usage'],
  ): ChatGenerationChunk {
    const finish_reason =
      stop === 'toolUse' ? 'tool_calls' : stop === 'maxTokens' ? 'length' : 'stop'
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk<StandardMessageStructure>({
        content: '',
        response_metadata: {
          model: modelId,
          model_name: modelId,
          model_provider: 'webllm',
          finish_reason,
          ...(usage && { webllm: usage.extra }),
        },
        ...(usage && {
          usage_metadata: {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
          },
        }),
      }),
      generationInfo: { finish_reason },
    })
  }

  function checkCalls(ordered: ToolCall[], stop: NonNullable<StreamChunk['stop']>): void {
    checkToolChoice(ordered, choice, stop, toolNames)
    if (stop === 'toolUse' && !ordered.length) throw new Error('WebLLM returned no tool calls.')
  }

  function* finishStream(
    stop: NonNullable<StreamChunk['stop']>,
    usage: StreamChunk['usage'],
    ordered: ToolCall[],
  ): Generator<ChatGenerationChunk> {
    if (ordered.length && stop !== 'maxTokens') {
      checkToolCallIds(ordered)
      const inputs = ordered.map(checkToolCall)
      if (deferred) yield* emitDeferred(true)
      else {
        for (const [index, call] of ordered.entries()) {
          call.id ??= crypto.randomUUID()
          const chunk = call.streamed
            ? toolCallIdChunk(call, call.id, inputs[index]!)
            : toolCallChunk(call)
          yield chunk
          signal?.throwIfAborted()
        }
      }
      stop = 'toolUse'
    } else if (deferred) yield* emitDeferred(false)
    yield finishChunk(stop, usage)
  }

  try {
    for await (const chunk of chunks) {
      signal?.throwIfAborted()
      if (hasContentAfterStop(stop, chunk))
        throw new Error('WebLLM returned content after its stop reason.')
      if (chunk.stop) stop = chunk.stop
      if (chunk.calls.length) block = undefined
      for (const delta of chunk.calls) {
        const call = recordToolDelta(delta)
        if (!streamTools) continue
        if (call.streamed) {
          yield toolCallDeltaChunk(call.index, delta, delta.name ? call.name : undefined)
          signal?.throwIfAborted()
          continue
        }
        call.pending.push({ delta, ...(delta.name && { name: call.name }) })
        while (true) {
          const ready = calls.get(nextToolCallIndex)
          if (!ready) break
          for (const pending of ready.pending) {
            yield toolCallDeltaChunk(ready.index, pending.delta, pending.name)
            signal?.throwIfAborted()
          }
          ready.pending = []
          ready.streamed = true
          nextToolCallIndex++
        }
      }
      for (const [type, field] of contentFields) {
        const value = chunk[field]
        if (!value) continue
        if (block !== type) {
          block = type
          index++
        }
        const contentChunk = makeContentChunk(type, value, index)
        if (deferred) deferred.push(contentChunk)
        else yield contentChunk
        signal?.throwIfAborted()
      }
      if (chunk.usage) usage = chunk.usage
    }
    signal?.throwIfAborted()
    if (!stop) throw new Error('WebLLM ended the stream without a stop reason.')
    throwIfCancelled(stop)
    const ordered = [...calls.values()].sort((left, right) => left.index - right.index)
    checkCalls(ordered, stop)
    if (stop === 'maxTokens' && streamTools)
      for (const call of ordered)
        for (const pending of call.pending) {
          yield toolCallDeltaChunk(call.index, pending.delta, pending.name)
          signal?.throwIfAborted()
        }
    yield* finishStream(stop, usage, ordered)
  } catch (error) {
    if (!signal?.aborted) yield* emitDeferred(false)
    throw error
  }
}

function makeContentChunk(
  type: 'text' | 'reasoning',
  value: string,
  index: number,
): ChatGenerationChunk {
  const content =
    type === 'reasoning' ? [{ type, reasoning: value, index }] : [{ type, text: value, index }]
  return new ChatGenerationChunk({
    text: type === 'text' ? value : '',
    message: new AIMessageChunk({ content }),
  })
}

function throwIfCancelled(stop: NonNullable<StreamChunk['stop']>): void {
  if (stop !== 'cancelled') return
  const error = new Error('WebLLM cancelled generation.')
  error.name = 'AbortError'
  throw stampRetryable(error, false)
}

function canStreamToolCalls(choice: ToolChoice | 'none' | undefined): boolean {
  return choice !== 'none' && (!choice || 'auto' in choice || 'any' in choice)
}

function toolCallDeltaChunk(
  callIndex: number,
  delta: ToolDelta,
  name?: string,
): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          type: 'tool_call_chunk',
          index: callIndex,
          ...(name && { name }),
          args: delta.input ?? '',
        },
      ],
    }),
  })
}

function toolCallChunk(call: ToolCall): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          type: 'tool_call_chunk',
          index: call.index,
          id: call.id ?? crypto.randomUUID(),
          name: call.name,
          args: call.input,
        },
      ],
    }),
  })
}

function toolCallIdChunk(
  call: ToolCall,
  id: string,
  args: Record<string, unknown>,
): ChatGenerationChunk {
  const message = new AIMessageChunk({
    content: '',
    tool_call_chunks: [{ type: 'tool_call_chunk', index: call.index, id }],
  })
  // The ID-only delta completes prior fragments; it is not a standalone empty call.
  message.tool_calls = [{ type: 'tool_call', id, name: call.name, args }]
  message.lc_kwargs.tool_calls = message.tool_calls
  return new ChatGenerationChunk({
    text: '',
    message,
  })
}

function checkToolCall(call: ToolCall): Record<string, unknown> {
  if (!call.name) throw new Error('WebLLM returned a tool call without a name.')
  const input = parseJSON(call.input)
  if (typeof input !== 'object' || input === null || Array.isArray(input))
    throw new Error('WebLLM tool arguments must be an object.')
  return input
}
