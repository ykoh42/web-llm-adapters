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
import { throwStreamError } from '../stream/failure.js'
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

interface FrameOptions {
  parseProtocol: boolean
  parseThinking: boolean
  thinkingPrefilled: boolean
  thinkingDisabled: boolean
  parseTools: boolean
  stopRequested: boolean
}

function createFrameParser(
  reader: ToolReader,
  {
    parseProtocol,
    parseThinking,
    thinkingPrefilled,
    thinkingDisabled,
    parseTools,
    stopRequested,
  }: FrameOptions,
) {
  const textMarkers: string[] = []
  if (parseThinking && !thinkingDisabled) textMarkers.push(thinkStart)
  if (parseTools) textMarkers.push(callStart)
  let mode: 'text' | 'thinking' | 'tool' =
    parseProtocol && parseThinking && thinkingPrefilled ? 'thinking' : 'text'
  let readingPrefix = thinkingDisabled
  let pending = ''
  let toolText = ''
  const output = new QwenOutput()
  let callIndex = 0
  let failure: { error: unknown } | undefined

  function readPrefix(): void {
    if (pending.startsWith(emptyThinking)) {
      pending = pending.slice(emptyThinking.length)
      if (parseProtocol && parseThinking) {
        output.add('reasoning', '\n\n')
        pending = '\n\n' + pending
      }
      readingPrefix = false
    } else if (!emptyThinking.startsWith(pending)) readingPrefix = false
  }

  function readTool(): boolean {
    const end = textEnd(pending, reader.markers ?? toolEndMarkers)
    const at = reader.end(pending, end)
    if (at === -1) {
      toolText += pending.slice(0, end)
      pending = pending.slice(end)
      return false
    }
    toolText += pending.slice(0, at)
    pending = pending.slice(at + callEnd.length)
    mode = 'text'
    if (!failure) {
      try {
        const parsed = reader.parse(toolText, callIndex)
        output.finishTool(parsed)
        callIndex += parsed.length
      } catch (error) {
        // Keep framing the received tail without publishing any failed call.
        failure = { error }
      }
    }
    toolText = ''
    return true
  }

  function readText(): boolean {
    if (mode === 'text' && (!parseProtocol || !textMarkers.length)) {
      output.add('text', pending)
      pending = ''
      return false
    }
    const markers = mode === 'thinking' ? thinkingMarkers : textMarkers
    const at = earliest(pending, markers)
    if (at === undefined) {
      const end = textEnd(pending, markers)
      const value = pending.slice(0, end)
      output.add(mode === 'thinking' ? 'reasoning' : 'text', value)
      pending = pending.slice(end)
      return false
    }
    const value = pending.slice(0, at.index)
    output.add(mode === 'thinking' ? 'reasoning' : 'text', value)
    pending = pending.slice(at.index + at.marker.length)
    if (mode === 'thinking') mode = 'text'
    else if (at.marker === thinkStart) mode = 'thinking'
    else {
      output.add('tool', callStart)
      mode = 'tool'
    }
    return true
  }

  function readPending(): void {
    while (pending && !readingPrefix) {
      if (mode === 'tool' ? !readTool() : !readText()) break
    }
  }

  function finish(stop: NonNullable<StreamChunk['stop']>): void {
    readingPrefix = false
    if (mode === 'tool') {
      if (stop !== 'maxTokens' && stop !== 'cancelled') throw new ModelError(reader.unclosedError())
      pending = ''
      toolText = ''
      mode = 'text'
    }
    output.add(mode === 'thinking' ? 'reasoning' : 'text', pending)
    pending = ''
    if (mode === 'thinking' && stop !== 'maxTokens' && stop !== 'cancelled' && !stopRequested)
      throw new ModelError('Qwen ended inside a thinking block.')
    mode = 'text'
  }

  return {
    output,
    get failure() {
      return failure
    },
    write(choice: ChatCompletionChunk['choices'][number] | undefined) {
      pending += choice?.delta.content ?? ''
      // WebLLM echoes this fixed block even when the decoder constrains the answer.
      if (readingPrefix) readPrefix()
      readPending()
      if (failure) throw failure.error
      const stop = stopReason(choice?.finish_reason)
      if (stop) finish(stop)
      return stop
    },
    flush() {
      output.add(mode === 'thinking' ? 'reasoning' : mode, pending)
    },
  }
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
  const parser = createFrameParser(reader, {
    parseProtocol,
    parseThinking,
    thinkingPrefilled,
    thinkingDisabled,
    parseTools,
    stopRequested,
  })
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      const parserChunk = onChunk && onChunkMayMutate ? snapshotChatChunk(chunk) : chunk
      const callbackResult = onChunk && callChunkCallback(onChunk, chunk, signal)
      if (callbackResult) await callbackResult
      if (signal?.aborted) return
      const choice = singleChoice(parserChunk.choices)
      const stop = parser.write(choice)
      yield* parser.output.read(stop)
      const metadata = metadataChunk(stop, parserChunk)
      if (metadata) yield metadata
    }
  } catch (error) {
    failure = parser.failure ?? { error }
  }
  if (signal?.aborted) return
  parser.flush()
  yield* parser.output.read('incomplete')
  throwStreamError(failure, signal)
}

function earliest(text: string, markers: string[]) {
  let found: { index: number; marker: string } | undefined
  for (const marker of markers) {
    const index = text.indexOf(marker)
    if (index !== -1 && (!found || index < found.index)) found = { index, marker }
  }
  return found
}

function metadataChunk(
  stop: StreamChunk['stop'],
  chunk: Pick<ChatCompletionChunk, 'usage'>,
): StreamChunk | undefined {
  if (!stop && !chunk.usage) return undefined
  return { text: '', reasoning: '', calls: noToolCalls, stop, usage: chunk.usage }
}
