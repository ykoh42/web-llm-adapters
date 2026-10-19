import { checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import type { ToolChoice } from '../../../../shared/web-llm/types.js'
import { AIMessageChunk, type StandardMessageStructure } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { StreamChunk, ToolDelta } from '../../../../shared/web-llm/stream/chunks.js'

/** Publish complete tool calls only; a truncated call must never run. */
export async function* readChunks(
  chunks: AsyncIterable<StreamChunk>,
  modelId: string,
  signal?: AbortSignal,
  choice?: ToolChoice | 'none',
): AsyncGenerator<ChatGenerationChunk> {
  const calls = new Map<number, ToolDelta>()
  let toolText = ''
  let stop: StreamChunk['stop']
  let usage: StreamChunk['usage']
  let block: 'text' | 'reasoning' | undefined
  let index = -1
  for await (const chunk of chunks) {
    signal?.throwIfAborted()
    if (chunk.stop) stop = chunk.stop
    toolText += chunk.toolText
    for (const delta of chunk.calls) {
      const previous = calls.get(delta.index)
      calls.set(delta.index, {
        index: delta.index,
        id: delta.id ?? previous?.id,
        name: (previous?.name ?? '') + (delta.name ?? ''),
        input: (previous?.input ?? '') + (delta.input ?? ''),
      })
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
    }
    if (chunk.usage) usage = chunk.usage
  }
  signal?.throwIfAborted()
  if (!stop) throw new Error('WebLLM ended the stream without a stop reason.')
  if (stop === 'cancelled') throw new Error('WebLLM cancelled generation.')
  checkToolChoice(
    Array.from(calls.values(), (call) => call.name),
    choice,
    stop,
  )
  if (calls.size && stop !== 'maxTokens') {
    for (const call of calls.values()) {
      if (!call.name) throw new Error('WebLLM returned a tool call without a name.')
      const input: unknown = JSON.parse(call.input ?? '')
      if (typeof input !== 'object' || input === null || Array.isArray(input))
        throw new Error('WebLLM tool arguments must be an object.')
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            {
              type: 'tool_call_chunk',
              index: call.index,
              id: call.id ?? crypto.randomUUID(),
              name: call.name,
              args: call.input!,
            },
          ],
        }),
      })
    }
    stop = 'toolUse'
  } else {
    if (stop === 'toolUse') throw new Error('WebLLM returned no tool calls.')
    if (toolText)
      yield new ChatGenerationChunk({
        text: toolText,
        message: new AIMessageChunk({ content: toolText }),
      })
  }
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
