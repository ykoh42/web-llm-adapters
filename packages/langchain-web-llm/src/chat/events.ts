import { AIMessageChunk, type UsageMetadata } from '@langchain/core/messages'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import type { ChatModelStreamEvent, FinishReason } from '@langchain/core/language_models/event'
import { parseJSON } from '../../../../shared/web-llm/json.js'

type EventBlock =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; reasoning: string }
  | { type: 'tool_call_chunk'; id?: string; name: string; args: string }
  | { type: 'tool_call'; id?: string; name: string; args: unknown }
  | { type: 'invalid_tool_call'; name: string; args: string }

/** Keep legacy chunks intact while emitting complete native LangChain events. */
export async function* readEvents(
  chunks: AsyncIterable<ChatGenerationChunk>,
  signal?: AbortSignal,
): AsyncGenerator<ChatModelStreamEvent> {
  const blocks = new Map<number, EventBlock>()
  const contentBlocks = new Map<number, number>()
  const toolBlocks = new Map<string, number>()
  let lastTextIndex: number | undefined
  let usage: UsageMetadata | undefined
  let responseMetadata: Record<string, unknown> = {}
  let finishReason: FinishReason | undefined
  let messageStarted = false
  let nextIndex = 0

  function nextBlockIndex(): number {
    // Blocks are never removed, so occupied indices stay occupied.
    while (blocks.has(nextIndex)) nextIndex++
    return nextIndex
  }

  function contentBlockIndex(index: unknown): number {
    if (typeof index !== 'number') return nextBlockIndex()
    const current = contentBlocks.get(index)
    if (current !== undefined) return current
    const mapped = blocks.has(index) ? nextBlockIndex() : index
    contentBlocks.set(index, mapped)
    return mapped
  }

  for await (const chunk of chunks) {
    signal?.throwIfAborted()
    const message = chunk.message
    if (!AIMessageChunk.isInstance(message))
      throw new Error('WebLLM returned a non-AI message chunk.')
    if (!messageStarted) {
      messageStarted = true
      yield { event: 'message-start', ...(message.id && { id: message.id }) }
      signal?.throwIfAborted()
    }

    const content = message.content
    if (typeof content === 'string' && content) {
      const index = lastTextIndex ?? nextBlockIndex()
      yield* appendContent(blocks, index, 'text', content, signal)
      lastTextIndex = index
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue
        if (part.type === 'text' && typeof part.text === 'string' && part.text) {
          const index = contentBlockIndex(part.index)
          yield* appendContent(blocks, index, 'text', part.text, signal)
          lastTextIndex = index
        } else if (
          part.type === 'reasoning' &&
          typeof part.reasoning === 'string' &&
          part.reasoning
        ) {
          yield* appendContent(
            blocks,
            contentBlockIndex(part.index),
            'reasoning',
            part.reasoning,
            signal,
          )
        }
      }
    }

    for (const toolChunk of message.tool_call_chunks ?? []) {
      const idKey = toolChunk.id ? `id:${toolChunk.id}` : undefined
      const indexKey = toolChunk.index !== undefined ? `index:${toolChunk.index}` : undefined
      if (!idKey && !indexKey)
        throw new Error('WebLLM returned a tool call without an index or ID.')
      let blockIndex = idKey ? toolBlocks.get(idKey) : undefined
      blockIndex ??= indexKey ? toolBlocks.get(indexKey) : undefined
      let block = blockIndex === undefined ? undefined : blocks.get(blockIndex)
      const isNew = !block
      if (!block) {
        blockIndex = nextBlockIndex()
        block = { type: 'tool_call_chunk', name: '', args: '' }
        blocks.set(blockIndex, block)
      }
      if (blockIndex === undefined) throw new Error('WebLLM returned a tool call without an index.')
      if (idKey) toolBlocks.set(idKey, blockIndex)
      if (indexKey) toolBlocks.set(indexKey, blockIndex)
      if (block.type !== 'tool_call_chunk')
        throw new Error(`WebLLM reused tool call ${idKey ?? indexKey} for another block.`)
      if (toolChunk.id && block.id && toolChunk.id !== block.id)
        throw new Error('WebLLM changed a tool call ID.')
      if (toolChunk.id && !block.id) block.id = toolChunk.id
      if (toolChunk.name) block.name = toolChunk.name
      if (toolChunk.args) block.args += toolChunk.args

      if (isNew) {
        yield {
          event: 'content-block-start',
          index: blockIndex,
          content: {
            type: 'tool_call_chunk',
            ...(block.id && { id: block.id }),
            ...(block.name && { name: block.name }),
            index: blockIndex,
          },
        }
        signal?.throwIfAborted()
      }
      if ((!isNew && toolChunk.name) || toolChunk.args || toolChunk.id) {
        yield {
          event: 'content-block-delta',
          index: blockIndex,
          delta: {
            type: 'block-delta',
            fields: {
              type: 'tool_call_chunk',
              ...(!isNew && toolChunk.name && { name: block.name }),
              ...(toolChunk.args && { args: block.args }),
              ...(toolChunk.id && { id: block.id }),
            },
          },
        }
        signal?.throwIfAborted()
      }
    }

    if (message.usage_metadata) usage = message.usage_metadata
    if (message.response_metadata.finish_reason) {
      finishReason = toFinishReason(message.response_metadata.finish_reason)
      responseMetadata = { ...message.response_metadata }
    }
  }

  signal?.throwIfAborted()
  if (!messageStarted || !finishReason)
    throw new Error('WebLLM ended the stream without a finish reason.')
  for (const [index, content] of blocks) {
    yield { event: 'content-block-finish', index, content: finishBlock(content, finishReason) }
    signal?.throwIfAborted()
  }
  if (usage) {
    yield { event: 'usage', usage }
    signal?.throwIfAborted()
  }
  yield {
    event: 'message-finish',
    reason: finishReason,
    ...(usage && { usage }),
    responseMetadata,
  }
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
