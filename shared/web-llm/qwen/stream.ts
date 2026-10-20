import { ModelError, type ToolSpec } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import {
  singleChoice,
  stopReason,
  type StreamChunk,
  type ToolDelta,
  type ChunkOptions,
} from '../stream/chunks.js'
import { textEnd } from '../gemma/markers.js'
import { parseJSON } from '../json.js'
import { QwenOutput } from './output.js'

const thinkStart = '<think>'
const thinkEnd = '</think>'
const callStart = '<tool_call>'
const callEnd = '</tool_call>'

/** Split Qwen reasoning and parse its tagged function calls across chunks. */
export async function* qwenChunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  {
    onChunk,
    signal,
    parseProtocol = true,
    parseThinking = true,
  }: ChunkOptions<ChatCompletionChunk> & {
    parseProtocol?: boolean
    parseThinking?: boolean
  } = {},
): AsyncGenerator<StreamChunk> {
  const textMarkers: string[] = []
  if (parseThinking) textMarkers.push(thinkStart)
  if (tools.length) textMarkers.push(callStart)
  let mode: 'text' | 'thinking' | 'tool' = 'text'
  let pending = ''
  let toolJson = ''
  const output = new QwenOutput()
  let callIndex = 0
  const quoted = { inside: false, escaped: false }
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      onChunk?.(chunk)
      if (signal?.aborted) return
      const choice = singleChoice(chunk.choices)
      pending += choice?.delta.content ?? ''
      const calls: ToolDelta[] = []

      while (pending) {
        if (mode === 'tool') {
          const end = textEnd(pending, [callEnd])
          const at = toolEnd(pending, end, quoted)
          if (at === -1) {
            output.add('tool', pending.slice(0, end))
            toolJson += pending.slice(0, end)
            pending = pending.slice(end)
            break
          }
          toolJson += pending.slice(0, at)
          output.add('tool', pending.slice(0, at) + callEnd)
          pending = pending.slice(at + callEnd.length)
          calls.push(parseCall(toolJson, tools, callIndex++))
          toolJson = ''
          mode = 'text'
          continue
        }

        if (mode === 'text' && (!parseProtocol || !textMarkers.length)) {
          output.add('text', pending)
          pending = ''
          break
        }

        const markers = mode === 'thinking' ? [thinkEnd] : textMarkers
        const at = earliest(pending, markers)
        if (at !== undefined) {
          const value = pending.slice(0, at.index)
          if (mode === 'thinking') output.add('reasoning', value)
          else output.add('text', value)
          pending = pending.slice(at.index + at.marker.length)
          if (mode === 'thinking') mode = 'text'
          else if (at.marker === thinkStart) mode = 'thinking'
          else {
            output.add('tool', callStart)
            mode = 'tool'
          }
          continue
        }
        const end = textEnd(pending, markers)
        const value = pending.slice(0, end)
        if (mode === 'thinking') output.add('reasoning', value)
        else output.add('text', value)
        pending = pending.slice(end)
        break
      }

      const stop = stopReason(choice?.finish_reason)
      if (stop) {
        if (mode === 'tool') {
          if (stop !== 'maxTokens' && stop !== 'cancelled') {
            const end = textEnd(pending, [callEnd])
            const content = toolJson + pending.slice(0, end)
            output.add('tool', pending)
            pending = ''
            calls.push(parseCall(content, tools, callIndex++))
          } else output.add('tool', pending)
          pending = ''
          toolJson = ''
          mode = 'text'
        }
        if (mode === 'thinking') {
          output.add('reasoning', pending)
          pending = ''
        } else {
          output.add('text', pending)
          pending = ''
        }
        mode = 'text'
      }
      yield* output.read(stop)
      if (calls.length || stop || chunk.usage)
        yield { text: '', reasoning: '', calls, stop, usage: chunk.usage }
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  output.add(mode === 'thinking' ? 'reasoning' : 'text', pending)
  yield* output.read('incomplete')
  if (failure && !signal?.aborted) throw failure.error
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

function earliest(text: string, markers: string[]) {
  let found: { index: number; marker: string } | undefined
  for (const marker of markers) {
    const index = text.indexOf(marker)
    if (index !== -1 && (!found || index < found.index)) found = { index, marker }
  }
  return found
}

function parseCall(text: string, tools: ToolSpec[], index: number): ToolDelta {
  const source = text.trim()
  let value: unknown
  try {
    value = parseJSON(source)
  } catch (cause) {
    if (cause instanceof ModelError) throw cause
    if (!source.endsWith('}')) throw new ModelError('Qwen returned invalid tool JSON.', { cause })
    try {
      value = parseJSON(source.slice(0, -1))
    } catch {
      throw new ModelError('Qwen returned invalid tool JSON.', { cause })
    }
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
