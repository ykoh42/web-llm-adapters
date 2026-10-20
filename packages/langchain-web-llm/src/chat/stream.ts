import { checkToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { parseJSON } from '../../../../shared/web-llm/json.js'
import type { ToolChoice } from '../../../../shared/web-llm/types.js'
import { AIMessageChunk, type StandardMessageStructure } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { ChatModelStreamEvent, FinishReason } from '@langchain/core/language_models/event'
import type { UsageMetadata } from '@langchain/core/messages'
import type { StreamChunk, ToolDelta } from '../../../../shared/web-llm/stream/chunks.js'

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
      const index = lastTextIndex ?? nextBlockIndex(blocks)
      yield* appendContent(blocks, index, 'text', content, signal)
      lastTextIndex = index
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue
        const index = typeof part.index === 'number' ? part.index : nextBlockIndex(blocks)
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
      const index = nextBlockIndex(blocks)
      yield {
        event: 'content-block-start',
        index,
        content: {
          type: 'tool_call_chunk',
          ...(toolChunk.id && { id: toolChunk.id }),
          ...(toolChunk.name && { name: toolChunk.name }),
          args: '',
          index,
        },
      }
      signal?.throwIfAborted()
      yield {
        event: 'content-block-delta',
        index,
        delta: {
          type: 'block-delta',
          fields: {
            type: 'tool_call_chunk',
            ...(toolChunk.id && { id: toolChunk.id }),
            ...(toolChunk.name && { name: toolChunk.name }),
            args: toolChunk.args ?? '',
          },
        },
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

function nextBlockIndex(blocks: Map<number, EventBlock>): number {
  let index = 0
  while (blocks.has(index)) index++
  return index
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function toFinishReason(reason: unknown): FinishReason {
  if (reason === 'tool_calls') return 'tool_use'
  if (reason === 'stop' || reason === 'length' || reason === 'content_filter') return reason
  throw new Error(`WebLLM returned an unsupported finish reason: ${String(reason)}.`)
}

/** Publish complete tool calls only; a truncated call must never run. */
export async function* readChunks(
  chunks: AsyncIterable<StreamChunk>,
  modelId: string,
  signal?: AbortSignal,
  choice?: ToolChoice | 'none',
): AsyncGenerator<ChatGenerationChunk> {
  const calls = new Map<number, ToolDelta>()
  let stop: StreamChunk['stop']
  let usage: StreamChunk['usage']
  let block: 'text' | 'reasoning' | undefined
  let index = -1
  for await (const chunk of chunks) {
    signal?.throwIfAborted()
    if (chunk.stop) stop = chunk.stop
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
      signal?.throwIfAborted()
    }
    if (chunk.usage) usage = chunk.usage
  }
  signal?.throwIfAborted()
  if (!stop) throw new Error('WebLLM ended the stream without a stop reason.')
  if (stop === 'cancelled') throw new Error('WebLLM cancelled generation.')
  const ordered = [...calls.values()].sort((left, right) => left.index - right.index)
  checkToolChoice(
    ordered.map((call) => call.name),
    choice,
    stop,
  )
  if (ordered.length && stop !== 'maxTokens') {
    for (const call of ordered) {
      yield toolCallChunk(call)
      signal?.throwIfAborted()
    }
    stop = 'toolUse'
  } else if (stop === 'toolUse') throw new Error('WebLLM returned no tool calls.')
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

function toolCallChunk(call: ToolDelta): ChatGenerationChunk {
  if (!call.name) throw new Error('WebLLM returned a tool call without a name.')
  const input = parseJSON(call.input ?? '')
  if (typeof input !== 'object' || input === null || Array.isArray(input))
    throw new Error('WebLLM tool arguments must be an object.')
  return new ChatGenerationChunk({
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
