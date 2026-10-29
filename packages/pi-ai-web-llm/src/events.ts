import type {
  Api,
  Model,
  AssistantMessage,
  AssistantMessageEventStream,
  ToolCall,
} from '@earendil-works/pi-ai'
import { parseStreamingJson } from '@earendil-works/pi-ai/utils/json-parse'
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai/utils/event-stream'
import { hasContentAfterStop, type StreamChunk } from '../../../shared/web-llm/stream/chunks.js'
import { errorText, isContextOverflow } from '../../../shared/web-llm/errors.js'
import { textBlocks } from './text.js'
import { toolCalls, type ToolCallUpdate, type ToolPolicy } from './tools.js'

export function responseStream(
  model: Model<Api>,
  generate: (message: AssistantMessage) => {
    chunks: AsyncIterable<StreamChunk>
    toolPolicy: ToolPolicy
  },
  signal?: AbortSignal,
  cleanup?: () => void,
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
    const partialCalls = new Map<number, { call: ToolCall; contentIndex: number }>()
    const pendingCalls = new Map<number, ToolCallUpdate>()
    let nextCallIndex = 0

    function appendCall(update: ToolCallUpdate) {
      let partial = partialCalls.get(update.index)
      if (!partial) {
        blocks.close()
        const call: ToolCall = {
          type: 'toolCall',
          id: update.id ?? '',
          name: update.name,
          arguments: {},
        }
        partial = { call, contentIndex: message.content.length }
        partialCalls.set(update.index, partial)
        message.content.push(call)
        stream.push({
          type: 'toolcall_start',
          contentIndex: partial.contentIndex,
          partial: message,
        })
      }
      if (update.id !== undefined) partial.call.id = update.id
      partial.call.name = update.name
      if (!update.argumentsDelta) return
      partial.call.arguments = partialArguments(update.argumentsText)
      stream.push({
        type: 'toolcall_delta',
        contentIndex: partial.contentIndex,
        delta: update.argumentsDelta,
        partial: message,
      })
    }

    function appendCalls(deltas: StreamChunk['calls']) {
      const updates = calls.append(deltas).sort((left, right) => left.index - right.index)
      for (const update of updates) {
        if (partialCalls.has(update.index)) {
          appendCall(update)
          continue
        }
        if (update.index > nextCallIndex) {
          pendingCalls.set(update.index, update)
          continue
        }
        appendCall(update)
        if (update.index !== nextCallIndex) continue
        nextCallIndex++
        while (pendingCalls.has(nextCallIndex)) {
          const pending = pendingCalls.get(nextCallIndex)!
          pendingCalls.delete(nextCallIndex)
          appendCall({ ...pending, argumentsDelta: pending.argumentsText })
          nextCallIndex++
        }
      }
    }

    function finishCalls(complete: ReturnType<typeof calls.finish>) {
      // Publish only complete, validated calls; partial arguments never become executable.
      for (const { index, call, delta } of complete) {
        signal?.throwIfAborted()
        let partial = partialCalls.get(index)
        if (!partial) {
          const contentIndex = message.content.length
          partial = { call, contentIndex }
          partialCalls.set(index, partial)
          message.content.push(call)
          stream.push({ type: 'toolcall_start', contentIndex, partial: message })
          if (delta) stream.push({ type: 'toolcall_delta', contentIndex, delta, partial: message })
        }
        Object.assign(partial.call, call)
        stream.push({
          type: 'toolcall_end',
          contentIndex: partial.contentIndex,
          toolCall: partial.call,
          partial: message,
        })
      }
    }

    function appendChunk(chunk: StreamChunk) {
      if (hasContentAfterStop(stop, chunk))
        throw new Error('WebLLM returned content after its stop reason.')
      if (chunk.usage) {
        message.usage.input = chunk.usage.prompt_tokens
        message.usage.output = chunk.usage.completion_tokens
        message.usage.totalTokens = chunk.usage.total_tokens
      }
      blocks.append('thinking', chunk.reasoning)
      blocks.append('text', chunk.text)
      appendCalls(chunk.calls)
      if (chunk.stop) stop = chunk.stop
    }

    function finishPartialCalls() {
      for (const [, update] of [...pendingCalls].sort(([left], [right]) => left - right))
        appendCall({ ...update, argumentsDelta: update.argumentsText })
      for (const { call } of partialCalls.values()) call.id ||= crypto.randomUUID()
    }

    try {
      signal?.throwIfAborted()
      const { chunks, toolPolicy } = generate(message)
      stream.push({ type: 'start', partial: message })
      for await (const chunk of chunks) {
        signal?.throwIfAborted()
        appendChunk(chunk)
      }
      signal?.throwIfAborted()
      if (!stop) throw new Error('WebLLM ended the stream without a stop reason.')
      if (stop === 'cancelled')
        throw new DOMException('WebLLM generation was interrupted.', 'AbortError')
      blocks.close()
      const complete = calls.finish(stop, toolPolicy)
      if (stop === 'maxTokens') finishPartialCalls()
      finishCalls(complete)
      message.stopReason = stop === 'maxTokens' ? 'length' : complete.length ? 'toolUse' : 'stop'
      stream.push({ type: 'done', reason: message.stopReason, message })
    } catch (error) {
      blocks.close()
      message.stopReason = signal?.aborted || isAbortError(error) ? 'aborted' : 'error'
      const text = errorText(error)
      // Pi detects context overflow from the error message.
      message.errorMessage =
        message.stopReason === 'error' && isContextOverflow(error)
          ? `Prompt is too long: ${text}`
          : text
      stream.push({ type: 'error', reason: message.stopReason, error: message })
    } finally {
      cleanup?.()
      stream.end()
    }
  }
}

function isAbortError(error: unknown): boolean {
  try {
    return (
      typeof error === 'object' && error !== null && 'name' in error && error.name === 'AbortError'
    )
  } catch {
    return false
  }
}

function partialArguments(source: string): ToolCall['arguments'] {
  const value: unknown = parseStreamingJson(source)
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as ToolCall['arguments'])
    : {}
}
