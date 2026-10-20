import { ModelError } from '../types.js'
import type { Completion } from '@mlc-ai/web-llm'
import { stopReason, type StreamChunk, type ChunkOptions } from '../stream/chunks.js'

export interface GemmaStreamOptions extends ChunkOptions<Completion> {
  prefix?: string
  stopRequested?: boolean
}
import { textEnd } from './markers.js'

type ThinkingChunk = Pick<StreamChunk, 'stop' | 'usage'> & {
  type: 'text' | 'reasoning'
  text: string
}

/** Split native thinking before the tool parser sees any generated text. */
export async function* splitThinking(
  chunks: AsyncIterable<Completion>,
  { onChunk, signal, prefix = '', stopRequested = false }: GemmaStreamOptions = {},
): AsyncGenerator<ThinkingChunk> {
  let pending = prefix
  let mode: 'text' | 'header' | 'thought' = 'text'
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
      onChunk?.(chunk)
      if (signal?.aborted) return
      const choice = chunk.choices[0]
      if (chunk.choices.length > 1 || (choice && choice.index !== 0))
        throw new ModelError('WebLLM adapters need one response per request.')
      received ||= Boolean(choice?.text)
      if (!choice || tool || (!choice.text && !choice.finish_reason)) {
        yield {
          type: 'text',
          text: choice?.text ?? '',
          stop: stopReason(choice?.finish_reason),
          usage: chunk.usage,
        }
        continue
      }
      pending += choice.text
      while (pending) {
        if (signal?.aborted) return
        let marker = '<|channel>'
        if (mode === 'header') marker = '\n'
        if (mode === 'thought') marker = '<channel|>'
        const at = pending.indexOf(marker)
        const call = mode === 'text' ? pending.indexOf('<|tool_call>') : -1
        if (call !== -1 && (at === -1 || call < at)) {
          if (call) yield piece(pending.slice(0, call), chunk.usage)
          yield piece(pending.slice(call), chunk.usage)
          pending = ''
          tool = true
          break
        }
        if (mode === 'header') {
          if (at === -1) break
          if (pending.slice(0, at) !== 'thought')
            throw new ModelError('Gemma returned an unknown thinking channel.')
        } else if (at !== -1) {
          if (at) yield piece(pending.slice(0, at), chunk.usage)
        } else {
          const markers = mode === 'text' ? [marker, '<|tool_call>'] : [marker]
          const end = textEnd(pending, markers)
          const text = pending.slice(0, end)
          pending = pending.slice(end)
          if (text) yield piece(text, chunk.usage)
          break
        }
        pending = pending.slice(at + marker.length)
        if (mode === 'text') {
          mode = 'header'
          // A channel boundary must not join text fragments into a tool marker.
          yield { type: 'reasoning', text: '', stop: undefined, usage: undefined }
        } else if (mode === 'header') mode = 'thought'
        else mode = 'text'
      }
      if (signal?.aborted) return
      if (choice.finish_reason) {
        // A native stop string may end a thought before its closing marker.
        if (mode !== 'text' && choice.finish_reason === 'stop' && !stopRequested)
          throw new ModelError('Gemma ended inside a thinking block.')
        if (pending || mode === 'header') yield piece(pending, chunk.usage)
        pending = ''
        mode = 'text'
        yield { type: 'text', text: '', stop: stopReason(choice.finish_reason), usage: chunk.usage }
      }
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  if (received && (pending || mode === 'header')) yield piece(pending)
  if (failure && !signal?.aborted) throw failure.error
}
