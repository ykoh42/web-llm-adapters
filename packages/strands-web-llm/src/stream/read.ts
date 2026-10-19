import { checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { parseJSON } from '../../../../shared/web-llm/json.js'
import { ModelError as ProtocolError, type ToolChoice } from '../../../../shared/web-llm/types.js'
import { ModelError, type ModelStreamEvent } from '@strands-agents/sdk'
import type { CompletionUsage } from '@mlc-ai/web-llm'
import type { StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'

type ToolCall = {
  id?: string
  name: string
  input: string
  sentInput: number
  started: boolean
  closed: boolean
}

export async function* readStream<T>(
  chunks: AsyncIterable<T>,
  decode: (chunk: T) => StreamChunk,
  signal?: AbortSignal,
  onChunk?: (chunk: T) => void,
  choice?: ToolChoice | 'none',
  toolNames?: readonly string[],
  progressiveTools = false,
): AsyncGenerator<ModelStreamEvent> {
  let block: 'textDelta' | 'reasoningContentDelta' | undefined
  let toolText = ''
  let stopReason: StreamChunk['stop']
  let usage: CompletionUsage | undefined
  const calls = new Map<number, ToolCall>()
  let activeToolIndex: number | undefined

  yield { type: 'modelMessageStartEvent', role: 'assistant' }
  if (signal?.aborted) {
    yield { type: 'modelMessageStopEvent', stopReason: 'cancelled' }
    return
  }

  try {
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
        const call = calls.get(delta.index) ?? {
          name: '',
          input: '',
          sentInput: 0,
          started: false,
          closed: false,
        }
        if (delta.id && !call.started) call.id = delta.id
        if (delta.name) call.name += delta.name
        if (delta.input) call.input += delta.input
        calls.set(delta.index, call)
      }

      // Only stream native, non-terminal deltas. WebLLM 0.2.86 exposes parsed
      // tool calls at generation end, so those are emitted by finishTool below.
      if (progressiveTools && !chunk.stop && !signal?.aborted) {
        if (
          activeToolIndex === undefined &&
          calls.size === 1 &&
          calls.has(0) &&
          calls.get(0)!.id &&
          toolNames?.includes(calls.get(0)!.name)
        ) {
          const call = calls.get(0)!
          if (call.name) {
            if (block) {
              yield { type: 'modelContentBlockStopEvent' }
              block = undefined
            }
            yield {
              type: 'modelContentBlockStartEvent',
              start: { type: 'toolUseStart', toolUseId: call.id!, name: call.name },
            }
            call.started = true
            activeToolIndex = 0
          }
        }

        const activeCall = activeToolIndex === undefined ? undefined : calls.get(activeToolIndex)
        if (activeCall?.started && activeCall.input.length > activeCall.sentInput) {
          yield {
            type: 'modelContentBlockDeltaEvent',
            delta: {
              type: 'toolUseInputDelta',
              input: activeCall.input.slice(activeCall.sentInput),
            },
          }
          activeCall.sentInput = activeCall.input.length
        }
      }
      if (chunk.stop) stopReason = chunk.stop
    }

    if (signal?.aborted) stopReason = 'cancelled'
    if (!stopReason) throw new ModelError('WebLLM ended the stream without a stop reason.')
    if (block) yield { type: 'modelContentBlockStopEvent' }

    const orderedCalls = [...calls.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, call]) => call)
    checkToolChoice(
      orderedCalls.map((call) => call.name),
      choice,
      stopReason,
    )
    const useTools = stopReason !== 'maxTokens' && stopReason !== 'cancelled' && calls.size > 0
    if (useTools) {
      for (const call of orderedCalls) yield* finishTool(call)
      stopReason = 'toolUse'
    } else {
      yield* closeStartedTools(calls, true)
      if (stopReason === 'toolUse') throw new ModelError('WebLLM returned no tool calls.')
      if (toolText) {
        yield { type: 'modelContentBlockStartEvent' }
        yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: toolText } }
        yield { type: 'modelContentBlockStopEvent' }
      }
    }
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
    yield { type: 'modelMessageStopEvent', stopReason }
  } catch (error) {
    yield* closeStartedTools(calls, true)
    throw error
  }
}

function* finishTool(call: ToolCall): Generator<ModelStreamEvent> {
  if (!call.name) throw new ModelError('WebLLM returned a tool call without a name.')
  let input: ReturnType<typeof parseJSON>
  try {
    input = parseJSON(call.input)
  } catch (cause) {
    if (call.started) yield* closeTool(call, true)
    if (cause instanceof ProtocolError) throw cause
    throw new ModelError('WebLLM returned invalid tool JSON.', { cause })
  }
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    if (call.started) yield* closeTool(call, true)
    throw new ModelError('WebLLM tool arguments must be an object.')
  }
  const id = (call.id ??= crypto.randomUUID())
  if (!call.started) {
    yield {
      type: 'modelContentBlockStartEvent',
      start: { type: 'toolUseStart', toolUseId: id, name: call.name },
    }
    call.started = true
  }
  if (call.input.length > call.sentInput) {
    yield {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input: call.input.slice(call.sentInput) },
    }
    call.sentInput = call.input.length
  }
  yield* closeTool(call)
}

function* closeStartedTools(
  calls: Map<number, ToolCall>,
  flushInput = false,
): Generator<ModelStreamEvent> {
  for (const [, call] of [...calls.entries()].sort(([left], [right]) => left - right)) {
    if (call.started) yield* closeTool(call, flushInput)
  }
}

function* closeTool(call: ToolCall, flushInput = false): Generator<ModelStreamEvent> {
  if (!call.started || call.closed) return
  if (flushInput && call.input.length > call.sentInput) {
    yield {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input: call.input.slice(call.sentInput) },
    }
    call.sentInput = call.input.length
  }
  call.closed = true
  yield { type: 'modelContentBlockStopEvent' }
}
