import { ModelError, type ModelStreamEvent, type StopReason } from '@strands-agents/sdk'
import type { ChatCompletionChunk, CompletionUsage } from '@mlc-ai/web-llm'

type ToolCall = { id: string; name: string; input: string }
/** Normalized chat chunks can carry a separate reasoning part. */
export type StreamChunk = ChatCompletionChunk & { reasoning?: string }

export async function* readStream(
  chunks: AsyncIterable<StreamChunk>,
  hasTools: boolean,
  signal?: AbortSignal,
  onChunk?: (chunk: ChatCompletionChunk) => void,
): AsyncGenerator<ModelStreamEvent> {
  let block: 'textDelta' | 'reasoningContentDelta' | undefined
  let toolText = ''
  let stopReason: StopReason | undefined
  let usage: CompletionUsage | undefined
  const calls = new Map<number, ToolCall>()

  yield { type: 'modelMessageStartEvent', role: 'assistant' }
  if (signal?.aborted) {
    yield { type: 'modelMessageStopEvent', stopReason: 'cancelled' }
    return
  }
  for await (const chunk of chunks) {
    if (signal?.aborted) break
    onChunk?.(chunk)
    if (chunk.usage) usage = chunk.usage
    const choice = chunk.choices[0]
    if (!choice) continue
    if (chunk.choices.length > 1 || choice.index !== 0) {
      throw new ModelError('Strands needs one response per request.')
    }
    const text = choice.delta.content
    // WebLLM's native tools API streams JSON before its parsed calls arrive.
    if (hasTools && text) toolText += text
    for (const [type, value] of [
      ['reasoningContentDelta', chunk.reasoning],
      ['textDelta', hasTools ? undefined : text],
    ] as const) {
      if (!value) continue
      if (block !== type) {
        if (block) yield { type: 'modelContentBlockStopEvent' }
        yield { type: 'modelContentBlockStartEvent' }
        block = type
      }
      yield { type: 'modelContentBlockDeltaEvent', delta: { type, text: value } }
    }
    for (const delta of choice.delta.tool_calls ?? []) {
      const call = calls.get(delta.index) ?? { id: crypto.randomUUID(), name: '', input: '' }
      if (delta.id) call.id = delta.id
      if (delta.function?.name) call.name += delta.function.name
      if (delta.function?.arguments) call.input += delta.function.arguments
      calls.set(delta.index, call)
    }
    switch (choice.finish_reason) {
      case 'stop':
        stopReason = 'endTurn'
        break
      case 'length':
        stopReason = 'maxTokens'
        break
      case 'abort':
        stopReason = 'cancelled'
        break
      case 'tool_calls':
        stopReason = 'toolUse'
        break
    }
  }

  if (signal?.aborted) stopReason = 'cancelled'
  if (!stopReason) throw new ModelError('WebLLM ended the stream without a stop reason.')
  if (block) yield { type: 'modelContentBlockStopEvent' }

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
