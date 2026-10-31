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
type ThinkingState = { pending: string; mode: ThinkingMode; tool: boolean; received: boolean }

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
  const state: ThinkingState = { pending: prefix, mode: 'text', tool: false, received: false }
  const chunkOptions = {
    onChunk,
    onChunkMayMutate,
    parseTools,
    signal,
    stopRequested,
  }
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      const keepReading = yield* readChunk(chunk, state, chunkOptions)
      if (!keepReading) return
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  if (state.received && (state.pending || state.mode === 'header'))
    yield thinkingPiece(state, state.pending)
  throwStreamError(failure, signal)
}

async function* readChunk(
  chunk: Completion,
  state: ThinkingState,
  {
    onChunk,
    onChunkMayMutate,
    parseTools,
    signal,
    stopRequested,
  }: GemmaStreamOptions & { parseTools: boolean },
): AsyncGenerator<ThinkingChunk, boolean> {
  if (signal?.aborted) return false
  const parserChunk = onChunk && onChunkMayMutate ? snapshotCompletion(chunk) : chunk
  const callbackResult = onChunk ? callChunkCallback(onChunk, chunk, signal) : undefined
  if (callbackResult) await callbackResult
  if (signal?.aborted) return false
  const choice = singleChoice(parserChunk.choices)
  state.received ||= Boolean(choice?.text)
  if (!choice || state.tool || (!choice.text && !choice.finish_reason)) {
    yield {
      type: 'text',
      text: choice?.text ?? '',
      stop: stopReason(choice?.finish_reason),
      usage: parserChunk.usage,
    }
    return true
  }
  state.pending += choice.text
  yield* readPending(state, parseTools, parserChunk.usage, signal)
  if (signal?.aborted) return false
  if (!choice.finish_reason) return true
  // A native stop string may end a thought before its closing marker.
  if (state.mode !== 'text' && choice.finish_reason === 'stop' && !stopRequested)
    throw new ModelError('Gemma ended inside a thinking block.')
  if (state.pending || state.mode === 'header')
    yield thinkingPiece(state, state.pending, parserChunk.usage)
  state.pending = ''
  state.mode = 'text'
  yield {
    type: 'text',
    text: '',
    stop: stopReason(choice.finish_reason),
    usage: parserChunk.usage,
  }
  return true
}

function* readPending(
  state: ThinkingState,
  parseTools: boolean,
  usage: StreamChunk['usage'],
  signal?: AbortSignal,
): Generator<ThinkingChunk> {
  while (state.pending && !state.tool) {
    if (signal?.aborted) return
    let keepParsing: boolean
    if (state.mode === 'text') keepParsing = yield* readText(state, parseTools, usage)
    else if (state.mode === 'header') keepParsing = readHeader(state)
    else keepParsing = yield* readThought(state, parseTools, usage)
    if (!keepParsing) return
  }
}

function* readText(
  state: ThinkingState,
  parseTools: boolean,
  usage: StreamChunk['usage'],
): Generator<ThinkingChunk, boolean> {
  const channel = state.pending.indexOf(channelStart)
  const call = parseTools ? state.pending.indexOf(toolStart) : -1
  if (call !== -1 && (channel === -1 || call < channel)) {
    if (call) yield thinkingPiece(state, state.pending.slice(0, call), usage)
    yield thinkingPiece(state, state.pending.slice(call), usage)
    state.pending = ''
    state.tool = true
    return false
  }
  if (channel === -1) {
    const end = textEnd(state.pending, partialMarkers('text', parseTools))
    const text = state.pending.slice(0, end)
    state.pending = state.pending.slice(end)
    if (text) yield thinkingPiece(state, text, usage)
    return false
  }
  if (channel) yield thinkingPiece(state, state.pending.slice(0, channel), usage)
  state.pending = state.pending.slice(channel + channelStart.length)
  state.mode = 'header'
  // A channel boundary must not join text fragments into a tool marker.
  yield { type: 'reasoning', text: '', stop: undefined, usage: undefined }
  return true
}

function readHeader(state: ThinkingState): boolean {
  const end = state.pending.indexOf('\n')
  if (end === -1) return false
  if (state.pending.slice(0, end) !== 'thought')
    throw new ModelError('Gemma returned an unknown thinking channel.')
  state.pending = state.pending.slice(end + 1)
  state.mode = 'thought'
  return true
}

function* readThought(
  state: ThinkingState,
  parseTools: boolean,
  usage: StreamChunk['usage'],
): Generator<ThinkingChunk, boolean> {
  const end = state.pending.indexOf(channelEnd)
  if (end === -1) {
    const textEndIndex = textEnd(state.pending, partialMarkers('thought', parseTools))
    const text = state.pending.slice(0, textEndIndex)
    state.pending = state.pending.slice(textEndIndex)
    if (text) yield thinkingPiece(state, text, usage)
    return false
  }
  if (end) yield thinkingPiece(state, state.pending.slice(0, end), usage)
  state.pending = state.pending.slice(end + channelEnd.length)
  state.mode = 'text'
  return true
}

function thinkingPiece(
  state: ThinkingState,
  text: string,
  usage?: StreamChunk['usage'],
): ThinkingChunk {
  return {
    type: state.mode === 'thought' ? 'reasoning' : 'text',
    text: state.mode === 'header' ? '<|channel>' + text : text,
    stop: undefined,
    usage,
  }
}

function partialMarkers(mode: 'text' | 'thought', parseTools: boolean): string[] {
  if (mode === 'thought') return thoughtEndMarkers
  return parseTools ? channelAndToolMarkers : channelStartMarkers
}
