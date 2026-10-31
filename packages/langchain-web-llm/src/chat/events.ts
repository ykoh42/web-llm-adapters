import { AIMessageChunk, type UsageMetadata } from '@langchain/core/messages'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import type { ChatModelStreamEvent, FinishReason } from '@langchain/core/language_models/event'
import { parseJSON } from '../../../../shared/web-llm/json.js'

type ToolChunk = NonNullable<AIMessageChunk['tool_call_chunks']>[number]
type ToolBlock = { type: 'tool_call_chunk'; id?: string; name: string; args: string }

type EventBlock =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; reasoning: string }
  | ToolBlock
  | { type: 'tool_call'; id?: string; name: string; args: unknown }
  | { type: 'invalid_tool_call'; name: string; args: string }

type EventState = {
  blocks: Map<number, EventBlock>
  contentBlocks: Map<number, number>
  toolBlocks: Map<string, number>
  lastTextIndex: number | undefined
  usage?: UsageMetadata
  responseMetadata: Record<string, unknown>
  finishReason?: FinishReason
  messageStarted: boolean
  nextIndex: number
}

/** Keep legacy chunks intact while emitting complete native LangChain events. */
export async function* readEvents(
  chunks: AsyncIterable<ChatGenerationChunk>,
  signal?: AbortSignal,
): AsyncGenerator<ChatModelStreamEvent> {
  const state: EventState = {
    blocks: new Map(),
    contentBlocks: new Map(),
    toolBlocks: new Map(),
    lastTextIndex: undefined,
    responseMetadata: {},
    messageStarted: false,
    nextIndex: 0,
  }

  for await (const chunk of chunks) {
    signal?.throwIfAborted()
    const message = chunk.message
    if (!AIMessageChunk.isInstance(message))
      throw new Error('WebLLM returned a non-AI message chunk.')
    if (!state.messageStarted) {
      state.messageStarted = true
      yield { event: 'message-start', ...(message.id && { id: message.id }) }
      signal?.throwIfAborted()
    }

    yield* readContentEvents(message, state, signal)
    yield* readToolEvents(message.tool_call_chunks ?? [], state, signal)

    if (message.usage_metadata) state.usage = message.usage_metadata
    if (message.response_metadata.finish_reason) {
      state.finishReason = toFinishReason(message.response_metadata.finish_reason)
      state.responseMetadata = { ...message.response_metadata }
    }
  }

  signal?.throwIfAborted()
  if (!state.messageStarted || !state.finishReason)
    throw new Error('WebLLM ended the stream without a finish reason.')
  for (const [index, content] of state.blocks) {
    yield {
      event: 'content-block-finish',
      index,
      content: finishBlock(content, state.finishReason),
    }
    signal?.throwIfAborted()
  }
  if (state.usage) {
    yield { event: 'usage', usage: state.usage }
    signal?.throwIfAborted()
  }
  yield {
    event: 'message-finish',
    reason: state.finishReason,
    ...(state.usage && { usage: state.usage }),
    responseMetadata: state.responseMetadata,
  }
}

function nextBlockIndex(state: EventState): number {
  // Blocks are never removed, so occupied indices stay occupied.
  while (state.blocks.has(state.nextIndex)) state.nextIndex++
  return state.nextIndex
}

function contentBlockIndex(state: EventState, index: unknown): number {
  if (typeof index !== 'number') return nextBlockIndex(state)
  const current = state.contentBlocks.get(index)
  if (current !== undefined) return current
  const mapped = state.blocks.has(index) ? nextBlockIndex(state) : index
  state.contentBlocks.set(index, mapped)
  return mapped
}

function toolBlockIndex(
  state: EventState,
  id: string | undefined,
  index: string | undefined,
): number {
  return (
    (id ? state.toolBlocks.get(id) : undefined) ??
    (index ? state.toolBlocks.get(index) : undefined) ??
    nextBlockIndex(state)
  )
}

function updateToolBlock(state: EventState, toolChunk: ToolChunk) {
  const idKey = toolChunk.id ? `id:${toolChunk.id}` : undefined
  const indexKey = toolChunk.index !== undefined ? `index:${toolChunk.index}` : undefined
  if (!idKey && !indexKey) throw new Error('WebLLM returned a tool call without an index or ID.')
  const blockIndex = toolBlockIndex(state, idKey, indexKey)
  let block = state.blocks.get(blockIndex)
  const isNew = !block
  if (!block) {
    block = { type: 'tool_call_chunk', name: '', args: '' }
    state.blocks.set(blockIndex, block)
  }
  if (idKey) state.toolBlocks.set(idKey, blockIndex)
  if (indexKey) state.toolBlocks.set(indexKey, blockIndex)
  if (block.type !== 'tool_call_chunk')
    throw new Error(`WebLLM reused tool call ${idKey ?? indexKey} for another block.`)
  if (toolChunk.id && block.id && toolChunk.id !== block.id)
    throw new Error('WebLLM changed a tool call ID.')
  if (toolChunk.id && !block.id) block.id = toolChunk.id
  if (toolChunk.name) block.name = toolChunk.name
  if (toolChunk.args) block.args += toolChunk.args
  return { blockIndex, block, isNew }
}

function* readContentEvents(
  message: AIMessageChunk,
  state: EventState,
  signal?: AbortSignal,
): Generator<ChatModelStreamEvent> {
  const content = message.content
  if (typeof content === 'string') {
    if (!content) return
    const index = state.lastTextIndex ?? nextBlockIndex(state)
    yield* appendContent(state.blocks, index, 'text', content, signal)
    state.lastTextIndex = index
    return
  }
  if (!Array.isArray(content)) return

  for (const part of content) {
    if (!isRecord(part)) continue
    if (part.type === 'text' && typeof part.text === 'string' && part.text) {
      const index = contentBlockIndex(state, part.index)
      yield* appendContent(state.blocks, index, 'text', part.text, signal)
      state.lastTextIndex = index
      continue
    }
    if (part.type === 'reasoning' && typeof part.reasoning === 'string' && part.reasoning) {
      state.lastTextIndex = undefined
      yield* appendContent(
        state.blocks,
        contentBlockIndex(state, part.index),
        'reasoning',
        part.reasoning,
        signal,
      )
    }
  }
}

function* readToolEvents(
  toolChunks: ToolChunk[],
  state: EventState,
  signal?: AbortSignal,
): Generator<ChatModelStreamEvent> {
  if (!toolChunks.length) return
  state.lastTextIndex = undefined
  for (const toolChunk of toolChunks) {
    const { blockIndex, block, isNew } = updateToolBlock(state, toolChunk)
    if (isNew) {
      yield toolStartEvent(blockIndex, block)
      signal?.throwIfAborted()
    }
    if ((!isNew && toolChunk.name) || toolChunk.args || toolChunk.id) {
      yield toolDeltaEvent(blockIndex, block, isNew, toolChunk)
      signal?.throwIfAborted()
    }
  }
}

function toolStartEvent(index: number, block: ToolBlock) {
  return {
    event: 'content-block-start',
    index,
    content: {
      type: 'tool_call_chunk',
      ...(block.id && { id: block.id }),
      ...(block.name && { name: block.name }),
      index,
    },
  } as const
}

function toolDeltaEvent(index: number, block: ToolBlock, isNew: boolean, toolChunk: ToolChunk) {
  return {
    event: 'content-block-delta',
    index,
    delta: {
      type: 'block-delta',
      fields: {
        type: 'tool_call_chunk',
        ...(!isNew && toolChunk.name && { name: block.name }),
        ...(toolChunk.args && { args: block.args }),
        ...(toolChunk.id && { id: block.id }),
      },
    },
  } as const
}

function finishBlock(block: EventBlock, reason: FinishReason): EventBlock {
  if (block.type !== 'tool_call_chunk') return { ...block }
  if (reason !== 'tool_use')
    return { type: 'invalid_tool_call', name: block.name, args: block.args }
  const args = parseJSON(block.args)
  if (typeof args !== 'object' || args === null || Array.isArray(args))
    throw new Error('WebLLM tool arguments must be an object.')
  return {
    type: 'tool_call',
    ...(block.id && { id: block.id }),
    name: block.name,
    args,
  }
}

function* appendContent(
  blocks: Map<number, EventBlock>,
  index: number,
  type: 'text' | 'reasoning',
  value: string,
  signal?: AbortSignal,
): Generator<ChatModelStreamEvent> {
  let block = blocks.get(index)
  if (block && block.type !== type)
    throw new Error(`WebLLM reused content block ${index} for ${type}.`)
  if (!block) {
    block = type === 'text' ? { type, text: '' } : { type, reasoning: '' }
    blocks.set(index, block)
    yield { event: 'content-block-start', index, content: { ...block } }
    signal?.throwIfAborted()
  }
  if (block.type === 'text') {
    block.text += value
    yield { event: 'content-block-delta', index, delta: { type: 'text-delta', text: value } }
  } else {
    block.reasoning += value
    yield {
      event: 'content-block-delta',
      index,
      delta: { type: 'reasoning-delta', reasoning: value },
    }
  }
  signal?.throwIfAborted()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function toFinishReason(reason: unknown): FinishReason {
  if (reason === 'tool_calls') return 'tool_use'
  if (reason === 'stop' || reason === 'length' || reason === 'content_filter') return reason
  throw new Error(`WebLLM returned an unsupported finish reason: ${String(reason)}.`)
}
