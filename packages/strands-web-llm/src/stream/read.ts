import { checkToolCallIds, checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { parseJSON } from '../../../../shared/web-llm/json.js'
import { ModelError as ProtocolError, type ToolChoice } from '../../../../shared/web-llm/types.js'
import { ModelError, type ModelStreamEvent } from '@strands-agents/sdk'
import type { CompletionUsage } from '@mlc-ai/web-llm'
import {
  hasContentAfterStop,
  type StreamChunk,
  type ToolDelta,
} from '../../../../shared/web-llm/stream/chunks.js'

type ToolCall = {
  id?: string
  name: string
  input: string[]
  sentParts: number
  started: boolean
  closed: boolean
}

type Deferred =
  | { type: 'textDelta' | 'reasoningContentDelta'; text: string }
  | { type: 'tool'; index: number }

const contentFields = [
  ['reasoningContentDelta', 'reasoning'],
  ['textDelta', 'text'],
] as const

export async function* readStream(
  chunks: AsyncIterable<StreamChunk>,
  signal?: AbortSignal,
  choice?: ToolChoice | 'none',
  {
    toolNames,
    progressiveTools = false,
    orderedTools = false,
  }: {
    toolNames?: readonly string[]
    progressiveTools?: boolean
    orderedTools?: boolean
  } = {},
): AsyncGenerator<ModelStreamEvent> {
  // Restricted calls must pass validation before any tool block is published.
  if (choice === 'none' || (choice && 'tool' in choice)) progressiveTools = false
  let block: 'textDelta' | 'reasoningContentDelta' | undefined
  let stopReason: StreamChunk['stop']
  let usage: CompletionUsage | undefined
  const calls = new Map<number, ToolCall>()
  let deferred: Deferred[] | undefined
  let deferredIndex = 0

  function recordToolDelta(delta: ToolDelta): void {
    if (orderedTools && !calls.has(delta.index)) {
      deferred ??= []
      deferred.push({ type: 'tool', index: delta.index })
    }
    const call = calls.get(delta.index) ?? {
      name: '',
      input: [],
      sentParts: 0,
      started: false,
      closed: false,
    }
    if (call.started && delta.name)
      throw new ModelError('WebLLM changed a tool name after streaming began.')
    if (delta.id && !call.started) call.id = delta.id
    if (delta.name) call.name += delta.name
    if (delta.input) call.input.push(delta.input)
    calls.set(delta.index, call)
  }

  function* closeBlock(): Generator<ModelStreamEvent> {
    if (!block) return
    block = undefined
    yield { type: 'modelContentBlockStopEvent' }
    signal?.throwIfAborted()
  }

  function* emitContent(
    type: 'textDelta' | 'reasoningContentDelta',
    text: string,
  ): Generator<ModelStreamEvent> {
    if (block !== type) {
      yield* closeBlock()
      block = type
      yield { type: 'modelContentBlockStartEvent' }
      signal?.throwIfAborted()
    }
    yield { type: 'modelContentBlockDeltaEvent', delta: { type, text } }
    signal?.throwIfAborted()
  }

  function* emitDeferred(includeTools: boolean): Generator<ModelStreamEvent> {
    while (deferred && deferredIndex < deferred.length) {
      const part = deferred[deferredIndex++]!
      if (part.type !== 'tool') {
        yield* emitContent(part.type, part.text)
        continue
      }
      if (!includeTools) continue
      yield* closeBlock()
      yield* finishTool(calls.get(part.index)!, signal)
    }
  }

  function* finishStream(
    stop: NonNullable<StreamChunk['stop']>,
    usage: CompletionUsage | undefined,
  ): Generator<ModelStreamEvent> {
    const orderedCalls = [...calls.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, call]) => call)
    checkToolChoice(orderedCalls, choice, stop, toolNames)
    if (stop !== 'maxTokens' && stop !== 'cancelled') checkToolCallIds(orderedCalls)
    const useTools = stop !== 'maxTokens' && stop !== 'cancelled' && calls.size > 0
    if (deferred) yield* emitDeferred(useTools)
    else if (useTools) {
      yield* closeBlock()
      for (const call of orderedCalls) {
        signal?.throwIfAborted()
        yield* finishTool(call, signal)
      }
    }
    if (useTools) stop = 'toolUse'
    else {
      if (!deferred) yield* closeBlock()
      yield* closeStartedTools(calls, true)
      if (stop === 'toolUse') throw new ModelError('WebLLM returned no tool calls.')
    }
    yield* closeBlock()
    signal?.throwIfAborted()
    if (usage) {
      const latency = usage.extra?.e2e_latency_s
      yield {
        type: 'modelMetadataEvent',
        usage: {
          inputTokens: usage.prompt_tokens,
          outputTokens: usage.completion_tokens,
          totalTokens: usage.total_tokens,
        },
        ...(typeof latency === 'number' &&
          latency >= 0 &&
          Number.isFinite(latency * 1000) && {
            metrics: { latencyMs: latency * 1000 },
          }),
        trace: usage.extra,
      }
    }
    signal?.throwIfAborted()
    yield { type: 'modelMessageStopEvent', stopReason: stop }
  }

  function* readContentDeltas(chunk: StreamChunk): Generator<ModelStreamEvent> {
    for (const [type, field] of contentFields) {
      const value = chunk[field]
      if (!value) continue
      if (deferred) {
        const previous = deferred.at(-1)
        if (previous?.type === type) previous.text += value
        else deferred.push({ type, text: value })
        continue
      }
      yield* emitContent(type, value)
    }
  }

  function readToolDeltas(deltas: readonly ToolDelta[]): void {
    for (const delta of deltas) recordToolDelta(delta)
  }

  function* emitProgressiveTool(stop: StreamChunk['stop']): Generator<ModelStreamEvent> {
    // WebLLM 0.2.86 exposes parsed tool calls at generation end, so only native
    // non-terminal deltas can be sent progressively.
    if (!progressiveTools || stop || signal?.aborted) return
    const firstCall = calls.get(0)
    if (canStartTool(calls, firstCall, toolNames)) {
      yield* closeBlock()
      firstCall.started = true
      yield {
        type: 'modelContentBlockStartEvent',
        start: { type: 'toolUseStart', toolUseId: firstCall.id, name: firstCall.name },
      }
      signal?.throwIfAborted()
    }
    if (firstCall?.started && firstCall.sentParts < firstCall.input.length) {
      const input = takeInput(firstCall)
      yield {
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'toolUseInputDelta', input },
      }
      signal?.throwIfAborted()
    }
  }

  function* readChunk(chunk: StreamChunk): Generator<ModelStreamEvent> {
    if (hasContentAfterStop(stopReason, chunk))
      throw new ModelError('WebLLM returned content after its stop reason.')
    if (chunk.usage) usage = chunk.usage
    yield* readContentDeltas(chunk)
    readToolDeltas(chunk.calls)
    yield* emitProgressiveTool(chunk.stop)
    if (chunk.stop) stopReason = chunk.stop
  }

  function* finishResponse(): Generator<ModelStreamEvent> {
    if (!stopReason) throw new ModelError('WebLLM ended the stream without a stop reason.')
    yield* finishStream(stopReason, usage)
  }

  yield { type: 'modelMessageStartEvent', role: 'assistant' }
  try {
    signal?.throwIfAborted()
    for await (const chunk of chunks) {
      signal?.throwIfAborted()
      yield* readChunk(chunk)
    }

    signal?.throwIfAborted()
    yield* finishResponse()
  } catch (error) {
    try {
      if (!signal?.aborted) yield* emitDeferred(false)
    } catch (cause) {
      if (!signal?.aborted) throw cause
    }
    if (block) yield { type: 'modelContentBlockStopEvent' }
    yield* closeStartedTools(calls, !signal?.aborted)
    if (signal?.aborted) {
      yield { type: 'modelMessageStopEvent', stopReason: 'cancelled' }
      return
    }
    throw error
  }
}

function canStartTool(
  calls: Map<number, ToolCall>,
  call: ToolCall | undefined,
  toolNames: readonly string[] | undefined,
): call is ToolCall & { id: string } {
  if (
    calls.size !== 1 ||
    !call?.id ||
    !call.name ||
    call.started ||
    !toolNames?.includes(call.name)
  )
    return false
  return !toolNames.some((name) => name !== call.name && name.startsWith(call.name))
}

function* finishTool(call: ToolCall, signal?: AbortSignal): Generator<ModelStreamEvent> {
  if (!call.name) throw new ModelError('WebLLM returned a tool call without a name.')
  let input: ReturnType<typeof parseJSON>
  const inputText = call.input.join('')
  try {
    input = parseJSON(inputText)
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
    call.started = true
    yield {
      type: 'modelContentBlockStartEvent',
      start: { type: 'toolUseStart', toolUseId: id, name: call.name },
    }
    signal?.throwIfAborted()
  }
  if (call.sentParts < call.input.length) {
    const input = takeInput(call)
    yield {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input },
    }
    signal?.throwIfAborted()
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

function takeInput(call: ToolCall): string {
  const input = call.input.slice(call.sentParts).join('')
  call.sentParts = call.input.length
  return input
}

function* closeTool(call: ToolCall, flushInput = false): Generator<ModelStreamEvent> {
  if (!call.started || call.closed) return
  if (flushInput && call.sentParts < call.input.length) {
    yield {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input: takeInput(call) },
    }
  }
  call.closed = true
  yield { type: 'modelContentBlockStopEvent' }
}
