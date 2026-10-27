import { ModelError, type ToolSpec } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import type { ToolDelta, ChunkOptions } from '../stream/chunks.js'
import { parseJSON } from '../json.js'
import { qwenFrames } from './frames.js'
import { isQwen35 } from '../models.js'
import { qwen35Chunks } from '../qwen35/stream.js'

const callEnd = '</tool_call>'

type QwenChunkOptions = ChunkOptions<ChatCompletionChunk> & {
  parseProtocol?: boolean
  thinkingPrefilled?: boolean
  thinkingDisabled?: boolean
  parseThinking?: boolean
  stopRequested?: boolean
}

export function qwenModelChunks(
  modelId: string,
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  options: Omit<QwenChunkOptions, 'parseThinking'> = {},
) {
  if (isQwen35(modelId)) return qwen35Chunks(chunks, tools, options)
  return qwenChunks(chunks, tools, { ...options, parseThinking: modelId.startsWith('Qwen3-') })
}

/** Qwen JSON closers are quote-aware, including across native chunk boundaries. */
export function qwenChunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  options: QwenChunkOptions = {},
) {
  const quoted = { inside: false, escaped: false }
  return qwenFrames(
    chunks,
    {
      end: (text, limit) => toolEnd(text, limit, quoted),
      parse: (text, index) => [parseCall(text, tools, index)],
      unclosedError: () => 'Qwen ended inside a tool call.',
    },
    { ...options, parseTools: tools.length > 0 },
  )
}

/** A closing tag inside a JSON string is data, including across chunk boundaries. */
function toolEnd(text: string, end: number, quoted: { inside: boolean; escaped: boolean }) {
  for (let index = 0; index < end; index++) {
    const character = text[index]
    if (quoted.inside) {
      if (quoted.escaped) quoted.escaped = false
      else if (character === '\\') quoted.escaped = true
      else if (character === '"') quoted.inside = false
    } else {
      if (text.startsWith(callEnd, index)) return index
      if (character === '"') quoted.inside = true
    }
  }
  return -1
}

function parseCall(text: string, tools: ToolSpec[], index: number): ToolDelta {
  const source = text.trim()
  let value: unknown
  try {
    value = parseJSON(source)
  } catch (cause) {
    if (cause instanceof ModelError) throw cause
    throw new ModelError('Qwen returned invalid tool JSON.', { cause })
  }
  if (!isToolCall(value)) throw new ModelError('Qwen returned a malformed tool call.')
  if (!tools.some((tool) => tool.name === value.name))
    throw new ModelError(`Qwen returned unknown tool "${value.name}".`)
  return {
    index,
    id: crypto.randomUUID(),
    name: value.name,
    input: JSON.stringify(value.arguments),
  }
}

function isToolCall(value: unknown): value is { name: string; arguments: Record<string, unknown> } {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const call = value as Record<string, unknown>
  return (
    typeof call.name === 'string' &&
    typeof call.arguments === 'object' &&
    call.arguments !== null &&
    !Array.isArray(call.arguments)
  )
}
