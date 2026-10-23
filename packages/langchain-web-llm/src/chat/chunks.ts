import { checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { parseJSON } from '../../../../shared/web-llm/json.js'
import type { ToolChoice } from '../../../../shared/web-llm/types.js'
import { AIMessageChunk, type StandardMessageStructure } from '@langchain/core/messages'
import { stampRetryable } from '@langchain/core/errors'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { StreamChunk, ToolDelta } from '../../../../shared/web-llm/stream/chunks.js'

interface ToolCall extends ToolDelta {
  name: string
  input: string
}

/** Publish complete tool calls only; a truncated call must never run. */
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
  for await (const chunk of chunks) {
    signal?.throwIfAborted()
    if (stop && (chunk.text || chunk.reasoning || chunk.calls.length || chunk.stop))
      throw new Error('WebLLM returned content after its stop reason.')
    if (chunk.stop) stop = chunk.stop
    for (const delta of chunk.calls) {
      let call = calls.get(delta.index)
      if (!call) {
        call = { index: delta.index, id: undefined, name: '', input: '' }
        calls.set(delta.index, call)
      }
      call.index = delta.index
      call.id = delta.id ?? call.id
      call.name += delta.name ?? ''
      call.input += delta.input ?? ''
    }
    for (const [type, value] of [
      ['reasoning', chunk.reasoning],
      ['text', chunk.text],
    ] as const) {
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
  checkToolChoice(
    ordered.map((call) => call.name),
    choice,
    stop,
    toolNames,
  )
  if (ordered.length && stop !== 'maxTokens') {
    for (const call of ordered) {
      yield toolCallChunk(call)
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

function toolCallChunk(call: ToolCall): ChatGenerationChunk {
  if (!call.name) throw new Error('WebLLM returned a tool call without a name.')
  const input = parseJSON(call.input)
  if (typeof input !== 'object' || input === null || Array.isArray(input))
    throw new Error('WebLLM tool arguments must be an object.')
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
