import type {
  Api,
  Model,
  AssistantMessage,
  AssistantMessageEventStream,
} from '@earendil-works/pi-ai'
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai/utils/event-stream'
import type { StreamChunk } from '../../../shared/web-llm/stream/chunks.js'
import { errorText } from '../../../shared/web-llm/errors.js'
import { textBlocks } from './text.js'
import { toolCalls } from './tools.js'

export function responseStream(
  model: Model<Api>,
  generate: (message: AssistantMessage) => {
    chunks: AsyncIterable<StreamChunk>
    toolNames: string[]
  },
  signal?: AbortSignal,
  choice?: 'none',
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream()
  const message: AssistantMessage = {
    role: 'assistant',
    api: model.api,
    provider: model.provider,
    model: model.id,
    content: [],
    stopReason: 'pending',
    timestamp: Date.now(),
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  }
  void run()
  return stream

  async function run() {
    let stop: StreamChunk['stop']
    const calls = toolCalls()
    const blocks = textBlocks(message, stream)

    try {
      signal?.throwIfAborted()
      const { chunks, toolNames } = generate(message)
      stream.push({ type: 'start', partial: message })
      for await (const chunk of chunks) {
        signal?.throwIfAborted()
        if (stop && (chunk.text || chunk.reasoning || chunk.calls.length || chunk.stop))
          throw new Error('WebLLM returned content after its stop reason.')
        if (chunk.usage) {
          message.usage.input = chunk.usage.prompt_tokens
          message.usage.output = chunk.usage.completion_tokens
          message.usage.totalTokens = chunk.usage.total_tokens
        }
        blocks.append('thinking', chunk.reasoning)
        blocks.append('text', chunk.text)
        calls.append(chunk.calls)
        if (chunk.stop) stop = chunk.stop
      }
      signal?.throwIfAborted()
      if (!stop) throw new Error('WebLLM ended the stream without a stop reason.')
      if (stop === 'cancelled')
        throw new DOMException('WebLLM generation was interrupted.', 'AbortError')
      blocks.close()
      const complete = calls.finish(stop, choice, toolNames)
      // Publish only complete, validated calls; truncated arguments never become executable.
      for (const { call, delta } of complete) {
        signal?.throwIfAborted()
        const contentIndex = message.content.length
        message.content.push({ ...call, arguments: {} })
        stream.push({ type: 'toolcall_start', contentIndex, partial: message })
        message.content[contentIndex] = call
        stream.push({ type: 'toolcall_delta', contentIndex, delta, partial: message })
        stream.push({ type: 'toolcall_end', contentIndex, toolCall: call, partial: message })
      }
      message.stopReason = stop === 'maxTokens' ? 'length' : complete.length ? 'toolUse' : 'stop'
      stream.push({ type: 'done', reason: message.stopReason, message })
    } catch (error) {
      blocks.close()
      message.stopReason =
        signal?.aborted || (error instanceof DOMException && error.name === 'AbortError')
          ? 'aborted'
          : 'error'
      message.errorMessage = errorText(error)
      stream.push({ type: 'error', reason: message.stopReason, error: message })
    } finally {
      stream.end()
    }
  }
}
