import { ModelError } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import {
  singleChoice,
  stopReason,
  snapshotChatChunk,
  type StreamChunk,
  type ToolDelta,
  type ChunkOptions,
  noToolCalls,
} from '../stream/chunks.js'
import { textEnd } from '../stream/markers.js'
import { QwenOutput } from './output.js'
import { callChunkCallback } from '../callback.js'

const thinkStart = '<think>'
const thinkEnd = '</think>'
const emptyThinking = '<think>\n\n</think>\n\n'
const callStart = '<tool_call>'
const callEnd = '</tool_call>'
const thinkingMarkers = [thinkEnd]
const toolEndMarkers = [callEnd]

interface ToolReader {
  end(text: string, limit: number): number
  parse(text: string, index: number): ToolDelta[]
  /** Retain partial markers needed by a stateful reader. */
  markers?: string[]
  /** Reject a normal stop when the tool call has no closing tag. */
  unclosedError(): string
}

/** Shared Qwen text/thinking framing; tool boundaries and syntax belong to each dialect. */
export async function* qwenFrames(
  chunks: AsyncIterable<ChatCompletionChunk>,
  reader: ToolReader,
  {
    onChunk,
    onChunkMayMutate = true,
    signal,
    parseProtocol = true,
    parseThinking = true,
    thinkingPrefilled = false,
    thinkingDisabled = false,
    parseTools,
    stopRequested = false,
  }: ChunkOptions<ChatCompletionChunk> & {
    parseProtocol?: boolean
    parseThinking?: boolean
    thinkingPrefilled?: boolean
    thinkingDisabled?: boolean
    parseTools: boolean
    stopRequested?: boolean
  },
): AsyncGenerator<StreamChunk> {
  const textMarkers: string[] = []
  if (parseThinking) textMarkers.push(thinkStart)
  if (parseTools) textMarkers.push(callStart)
  let mode: 'text' | 'thinking' | 'tool' =
    parseProtocol && parseThinking && thinkingPrefilled ? 'thinking' : 'text'
  let readingPrefix = !parseProtocol && thinkingDisabled
  let pending = ''
  let toolText = ''
  const output = new QwenOutput()
  let callIndex = 0
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      const parserChunk = onChunk && onChunkMayMutate ? snapshotChatChunk(chunk) : chunk
      const callbackResult = onChunk ? callChunkCallback(onChunk, chunk, signal) : undefined
      if (callbackResult) await callbackResult
      if (signal?.aborted) return
      const choice = singleChoice(parserChunk.choices)
      pending += choice?.delta.content ?? ''
      let calls: ToolDelta[] | undefined

      // WebLLM echoes this fixed block even when the decoder constrains the answer.
      if (readingPrefix) {
        if (pending.startsWith(emptyThinking)) {
          pending = pending.slice(emptyThinking.length)
          readingPrefix = false
        } else if (!emptyThinking.startsWith(pending)) readingPrefix = false
      }

      while (pending && !readingPrefix) {
        if (mode === 'tool') {
          const end = textEnd(pending, reader.markers ?? toolEndMarkers)
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
          mode = 'text'
          if (!failure) {
            try {
              const parsed = reader.parse(toolText, callIndex)
              if (parsed.length) {
                calls ??= []
                calls.push(...parsed)
                callIndex += parsed.length
              }
            } catch (error) {
              // Keep framing the received tail without publishing any failed call.
              failure = { error }
            }
          }
          toolText = ''
          continue
        }

        if (mode === 'text' && (!parseProtocol || !textMarkers.length)) {
          output.add('text', pending)
          pending = ''
          break
        }

        const markers = mode === 'thinking' ? thinkingMarkers : textMarkers
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

      if (failure) throw failure.error
      const stop = stopReason(choice?.finish_reason)
      if (stop) {
        readingPrefix = false
        if (mode === 'tool') {
          if (stop !== 'maxTokens' && stop !== 'cancelled') {
            throw new ModelError(reader.unclosedError())
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
      if (calls?.length || stop || parserChunk.usage)
        yield {
          text: '',
          reasoning: '',
          calls: calls ?? noToolCalls,
          stop,
          usage: parserChunk.usage,
        }
    }
  } catch (error) {
    failure ??= { error }
  }
  if (signal?.aborted) return
  output.add(mode === 'thinking' ? 'reasoning' : mode, pending)
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
