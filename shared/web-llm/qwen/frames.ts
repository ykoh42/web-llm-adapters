import { ModelError } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import {
  singleChoice,
  stopReason,
  type StreamChunk,
  type ToolDelta,
  type ChunkOptions,
} from '../stream/chunks.js'
import { textEnd } from '../stream/markers.js'
import { QwenOutput } from './output.js'

const thinkStart = '<think>'
const thinkEnd = '</think>'
const callStart = '<tool_call>'
const callEnd = '</tool_call>'

interface ToolReader {
  end(text: string, limit: number): number
  parse(text: string, index: number): ToolDelta[]
  /** Retain partial markers needed by a stateful reader. */
  markers?: string[]
  /** XML requires a closing tag; JSON also accepts a complete object at a normal stop. */
  unclosedError?: string
}

/** Shared Qwen text/thinking framing; tool boundaries and syntax belong to each dialect. */
export async function* qwenFrames(
  chunks: AsyncIterable<ChatCompletionChunk>,
  reader: ToolReader,
  {
    onChunk,
    signal,
    parseProtocol = true,
    parseThinking = true,
    thinkingPrefilled = false,
    parseTools,
    stopRequested = false,
  }: ChunkOptions<ChatCompletionChunk> & {
    parseProtocol?: boolean
    parseThinking?: boolean
    thinkingPrefilled?: boolean
    parseTools: boolean
    stopRequested?: boolean
  },
): AsyncGenerator<StreamChunk> {
  const textMarkers: string[] = []
  if (parseThinking) textMarkers.push(thinkStart)
  if (parseTools) textMarkers.push(callStart)
  let mode: 'text' | 'thinking' | 'tool' =
    parseProtocol && parseThinking && thinkingPrefilled ? 'thinking' : 'text'
  let pending = ''
  let toolText = ''
  const output = new QwenOutput()
  let callIndex = 0
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
          const end = textEnd(pending, reader.markers ?? [callEnd])
          const at = reader.end(pending, end)
          if (at === -1) {
            output.add('tool', pending.slice(0, end))
            toolText += pending.slice(0, end)
            pending = pending.slice(end)
            break
          }
          toolText += pending.slice(0, at)
          output.add('tool', pending.slice(0, at) + callEnd)
          pending = pending.slice(at + callEnd.length)
          const parsed = reader.parse(toolText, callIndex)
          calls.push(...parsed)
          callIndex += parsed.length
          toolText = ''
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
            if (reader.unclosedError) throw new ModelError(reader.unclosedError)
            const end = textEnd(pending, [callEnd])
            const content = toolText + pending.slice(0, end)
            output.add('tool', pending)
            pending = ''
            const parsed = reader.parse(content, callIndex)
            calls.push(...parsed)
            callIndex += parsed.length
          } else output.add('tool', pending)
          pending = ''
          toolText = ''
          mode = 'text'
        }
        if (mode === 'thinking') {
          output.add('reasoning', pending)
          pending = ''
          if (stop !== 'maxTokens' && stop !== 'cancelled' && !stopRequested)
            throw new ModelError('Qwen ended inside a thinking block.')
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

function earliest(text: string, markers: string[]) {
  let found: { index: number; marker: string } | undefined
  for (const marker of markers) {
    const index = text.indexOf(marker)
    if (index !== -1 && (!found || index < found.index)) found = { index, marker }
  }
  return found
}
