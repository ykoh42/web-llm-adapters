import { checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import type { ToolChoice } from '../../../../shared/web-llm/types.js'
import { ModelError, type ModelStreamEvent } from '@strands-agents/sdk'
import type { CompletionUsage } from '@mlc-ai/web-llm'
import type { StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'

type ToolCall = { id: string; name: string; input: string }

export async function* readStream<T>(
  chunks: AsyncIterable<T>,
  decode: (chunk: T) => StreamChunk,
  signal?: AbortSignal,
  onChunk?: (chunk: T) => void,
  choice?: ToolChoice | 'none',
): AsyncGenerator<ModelStreamEvent> {
  let block: 'textDelta' | 'reasoningContentDelta' | undefined
  let toolText = ''
  let stopReason: StreamChunk['stop']
  let usage: CompletionUsage | undefined
  const calls = new Map<number, ToolCall>()

  yield { type: 'modelMessageStartEvent', role: 'assistant' }
  if (signal?.aborted) {
    yield { type: 'modelMessageStopEvent', stopReason: 'cancelled' }
    return
  }
  for await (const raw of chunks) {
    if (signal?.aborted) break
    onChunk?.(raw)
    const chunk = decode(raw)
    if (chunk.usage) usage = chunk.usage
    toolText += chunk.toolText
    for (const [type, value] of [
      ['reasoningContentDelta', chunk.reasoning],
      ['textDelta', chunk.text],
    ] as const) {
      if (!value) continue
      if (block !== type) {
        if (block) yield { type: 'modelContentBlockStopEvent' }
        yield { type: 'modelContentBlockStartEvent' }
        block = type
      }
      yield { type: 'modelContentBlockDeltaEvent', delta: { type, text: value } }
    }
    for (const delta of chunk.calls) {
      const call = calls.get(delta.index) ?? { id: crypto.randomUUID(), name: '', input: '' }
      if (delta.id) call.id = delta.id
      if (delta.name) call.name += delta.name
      if (delta.input) call.input += delta.input
      calls.set(delta.index, call)
    }
    if (chunk.stop) stopReason = chunk.stop
  }

  if (signal?.aborted) stopReason = 'cancelled'
  if (!stopReason) throw new ModelError('WebLLM ended the stream without a stop reason.')
  if (block) yield { type: 'modelContentBlockStopEvent' }

  checkToolChoice(
    Array.from(calls.values(), (call) => call.name),
    choice,
    stopReason,
  )
  const useTools = stopReason !== 'maxTokens' && stopReason !== 'cancelled' && calls.size > 0
  if (useTools) {
    for (const call of calls.values()) yield* toolEvents(call)
    stopReason = 'toolUse'
  } else {
    if (stopReason === 'toolUse') throw new ModelError('WebLLM returned no tool calls.')
    if (toolText) {
      yield { type: 'modelContentBlockStartEvent' }
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: toolText } }
      yield { type: 'modelContentBlockStopEvent' }
    }
  }
  yield { type: 'modelMessageStopEvent', stopReason }
  if (usage) {
    yield {
      type: 'modelMetadataEvent',
      usage: {
        inputTokens: usage.prompt_tokens,
        outputTokens: usage.completion_tokens,
        totalTokens: usage.total_tokens,
      },
      trace: usage.extra,
    }
  }
}

function* toolEvents(call: ToolCall): Generator<ModelStreamEvent> {
  if (!call.name) throw new ModelError('WebLLM returned a tool call without a name.')
  try {
    JSON.parse(call.input)
  } catch (cause) {
    throw new ModelError('WebLLM returned invalid tool JSON.', { cause })
  }
  yield {
    type: 'modelContentBlockStartEvent',
    start: { type: 'toolUseStart', toolUseId: call.id, name: call.name },
  }
  yield {
    type: 'modelContentBlockDeltaEvent',
    delta: { type: 'toolUseInputDelta', input: call.input },
  }
  yield { type: 'modelContentBlockStopEvent' }
}
