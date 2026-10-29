import { ModelError } from '../types.js'
import type { Completion } from '@mlc-ai/web-llm'
import {
  singleChoice,
  snapshotCompletion,
  stopReason,
  type StreamChunk,
  type ChunkOptions,
} from '../stream/chunks.js'
import { toolStart } from './markers.js'
import { textEnd } from '../stream/markers.js'
import { callChunkCallback } from '../callback.js'
import { throwStreamError } from '../stream/failure.js'

const channelStart = '<|channel>'
const channelEnd = '<channel|>'
const channelStartMarkers = [channelStart]
const channelAndToolMarkers = [channelStart, toolStart]
const thoughtEndMarkers = [channelEnd]

export interface GemmaStreamOptions extends ChunkOptions<Completion> {
  prefix?: string
  stopRequested?: boolean
}

type ThinkingMode = 'text' | 'header' | 'thought'

type ThinkingChunk = Pick<StreamChunk, 'stop' | 'usage'> & {
  type: 'text' | 'reasoning'
  text: string
}

/** Split native thinking before the tool parser sees any generated text. */
export async function* splitThinking(
  chunks: AsyncIterable<Completion>,
  {
    onChunk,
    onChunkMayMutate = true,
    signal,
    prefix = '',
    stopRequested = false,
    parseTools,
  }: GemmaStreamOptions & { parseTools: boolean },
): AsyncGenerator<ThinkingChunk> {
  let pending = prefix
  let mode: ThinkingMode = 'text'
  let tool = false
  let received = false
  function piece(text: string, usage?: StreamChunk['usage']): ThinkingChunk {
    return {
      type: mode === 'thought' ? 'reasoning' : 'text',
      text: mode === 'header' ? '<|channel>' + text : text,
      stop: undefined,
      usage,
    }
  }
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      const parserChunk = onChunk && onChunkMayMutate ? snapshotCompletion(chunk) : chunk
      const callbackResult = onChunk ? callChunkCallback(onChunk, chunk, signal) : undefined
      if (callbackResult) await callbackResult
      if (signal?.aborted) return
      const choice = singleChoice(parserChunk.choices)
      received ||= Boolean(choice?.text)
      if (!choice || tool || (!choice.text && !choice.finish_reason)) {
        yield {
          type: 'text',
          text: choice?.text ?? '',
          stop: stopReason(choice?.finish_reason),
          usage: parserChunk.usage,
        }
        continue
      }
      pending += choice.text
      while (pending) {
        if (signal?.aborted) return
        const marker = nextMarker(mode)
        const at = pending.indexOf(marker)
        const call = parseTools && mode === 'text' ? pending.indexOf(toolStart) : -1
        if (call !== -1 && (at === -1 || call < at)) {
          if (call) yield piece(pending.slice(0, call), parserChunk.usage)
          yield piece(pending.slice(call), parserChunk.usage)
          pending = ''
          tool = true
          break
        }
        if (at === -1) {
          if (mode === 'header') break
          const end = textEnd(pending, partialMarkers(mode, parseTools))
          const text = pending.slice(0, end)
          pending = pending.slice(end)
          if (text) yield piece(text, parserChunk.usage)
          break
        }
        if (mode === 'header') {
          if (pending.slice(0, at) !== 'thought')
            throw new ModelError('Gemma returned an unknown thinking channel.')
        } else if (at) yield piece(pending.slice(0, at), parserChunk.usage)
        pending = pending.slice(at + marker.length)
        if (mode === 'text') {
          mode = 'header'
          // A channel boundary must not join text fragments into a tool marker.
          yield { type: 'reasoning', text: '', stop: undefined, usage: undefined }
        } else if (mode === 'header') mode = 'thought'
        else mode = 'text'
      }
      if (signal?.aborted) return
      if (!choice.finish_reason) continue
      // A native stop string may end a thought before its closing marker.
      if (mode !== 'text' && choice.finish_reason === 'stop' && !stopRequested)
        throw new ModelError('Gemma ended inside a thinking block.')
      if (pending || mode === 'header') yield piece(pending, parserChunk.usage)
      pending = ''
      mode = 'text'
      yield {
        type: 'text',
        text: '',
        stop: stopReason(choice.finish_reason),
        usage: parserChunk.usage,
      }
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  if (received && (pending || mode === 'header')) yield piece(pending)
  throwStreamError(failure, signal)
}

function nextMarker(mode: ThinkingMode): string {
  if (mode === 'header') return '\n'
  return mode === 'thought' ? channelEnd : channelStart
}

function partialMarkers(mode: 'text' | 'thought', parseTools: boolean): string[] {
  if (mode === 'thought') return thoughtEndMarkers
  return parseTools ? channelAndToolMarkers : channelStartMarkers
}
