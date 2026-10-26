import { checkToolCallIds, checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { parseJSON } from '../../../../shared/web-llm/json.js'
import type { ToolChoice } from '../../../../shared/web-llm/types.js'
import { AIMessageChunk, type StandardMessageStructure } from '@langchain/core/messages'
import { stampRetryable } from '@langchain/core/errors'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { StreamChunk, ToolDelta } from '../../../../shared/web-llm/stream/chunks.js'

interface ToolCall extends ToolDelta {
  readonly index: number
  name: string
  input: string
  pending: { delta: ToolDelta; name?: string }[]
  streamed: boolean
}

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
  for await (const chunk of chunks) {
    signal?.throwIfAborted()
    if (stop && (chunk.text || chunk.reasoning || chunk.calls.length || chunk.stop))
      throw new Error('WebLLM returned content after its stop reason.')
    if (chunk.stop) stop = chunk.stop
    for (const delta of chunk.calls) {
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
      }
      if (delta.id) {
        if (call.id && delta.id !== call.id) throw new Error('WebLLM changed a tool call ID.')
        call.id = delta.id
      }
      call.name += delta.name ?? ''
      call.input += delta.input ?? ''
      if (canStreamToolCalls(choice)) {
        if (call.streamed) {
          yield toolCallDeltaChunk(call.index, delta, delta.name ? call.name : undefined)
          signal?.throwIfAborted()
        } else {
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
      }
    }
    for (const [type, field] of contentFields) {
      const value = chunk[field]
      if (!value) continue
      if (block !== type) {
        block = type
        index++
      }
      const content =
        type === 'reasoning' ? [{ type, reasoning: value, index }] : [{ type, text: value, index }]
      yield new ChatGenerationChunk({
        text: type === 'text' ? value : '',
        message: new AIMessageChunk({ content }),
      })
      signal?.throwIfAborted()
    }
    if (chunk.usage) usage = chunk.usage
  }
  signal?.throwIfAborted()
  if (!stop) throw new Error('WebLLM ended the stream without a stop reason.')
  if (stop === 'cancelled') {
    const error = new Error('WebLLM cancelled generation.')
    error.name = 'AbortError'
    throw stampRetryable(error, false)
  }
  const ordered = [...calls.values()].sort((left, right) => left.index - right.index)
  checkToolChoice(ordered, choice, stop, toolNames)
  if (ordered.length && stop !== 'maxTokens') {
    checkToolCallIds(ordered)
    const inputs = ordered.map(checkToolCall)
    for (const [index, call] of ordered.entries()) {
      call.id ??= crypto.randomUUID()
      if (!call.streamed) yield toolCallChunk(call)
      else yield toolCallIdChunk(call, call.id, inputs[index]!)
      signal?.throwIfAborted()
    }
    stop = 'toolUse'
  } else if (stop === 'toolUse') throw new Error('WebLLM returned no tool calls.')
  const finish_reason = stop === 'toolUse' ? 'tool_calls' : stop === 'maxTokens' ? 'length' : 'stop'
  yield new ChatGenerationChunk({
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
