import { AIMessageChunk, type UsageMetadata } from '@langchain/core/messages'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import type { ChatModelStreamEvent, FinishReason } from '@langchain/core/language_models/event'

type EventBlock =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; reasoning: string }
  | { type: 'tool_call'; id?: string; name: string; args: unknown }

/** Keep legacy chunks intact while emitting complete native LangChain events. */
export async function* readEvents(
  chunks: AsyncIterable<ChatGenerationChunk>,
  signal?: AbortSignal,
): AsyncGenerator<ChatModelStreamEvent> {
  const blocks = new Map<number, EventBlock>()
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
        const index = typeof part.index === 'number' ? part.index : nextBlockIndex()
        if (part.type === 'text' && typeof part.text === 'string' && part.text) {
          yield* appendContent(blocks, index, 'text', part.text, signal)
          lastTextIndex = index
        } else if (
          part.type === 'reasoning' &&
          typeof part.reasoning === 'string' &&
          part.reasoning
        ) {
          yield* appendContent(blocks, index, 'reasoning', part.reasoning, signal)
        }
      }
    }

    for (const toolChunk of message.tool_call_chunks ?? []) {
      const call = message.tool_calls?.find((item) => item.id === toolChunk.id)
      if (!call) throw new Error('WebLLM returned an invalid tool call.')
      const index = nextBlockIndex()
      yield {
        event: 'content-block-start',
        index,
        content: {
          type: 'tool_call_chunk',
          ...(toolChunk.id && { id: toolChunk.id }),
          ...(toolChunk.name && { name: toolChunk.name }),
          index,
        },
      }
      signal?.throwIfAborted()
      yield {
        event: 'content-block-delta',
        index,
        delta: { type: 'tool-call-delta', args: toolChunk.args ?? '' },
      }
      signal?.throwIfAborted()
      blocks.set(index, {
        type: 'tool_call',
        ...(call.id && { id: call.id }),
        name: call.name,
        args: call.args,
      })
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
    yield { event: 'content-block-finish', index, content: { ...content } }
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
