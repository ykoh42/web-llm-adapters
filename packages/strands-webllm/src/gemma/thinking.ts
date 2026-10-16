import { ModelError } from '@strands-agents/sdk'
import type { Completion } from '@mlc-ai/web-llm'

type ThinkingChunk = Completion & { reasoning?: string }

/** Split native thinking before the tool parser sees any generated text. */
export async function* splitThinking(
  chunks: AsyncIterable<Completion>,
  onChunk?: (chunk: Completion) => void,
  prefix = '',
): AsyncGenerator<ThinkingChunk> {
  let pending = prefix
  let mode: 'text' | 'header' | 'thought' = 'text'
  let tool = false
  for await (const chunk of chunks) {
    onChunk?.(chunk)
    const choice = chunk.choices[0]
    if (chunk.choices.length > 1 || (choice && choice.index !== 0))
      throw new ModelError('Strands needs one response per request.')
    if (!choice) {
      yield chunk
      continue
    }
    if (tool) {
      yield chunk
      continue
    }
    function piece(text: string): ThinkingChunk {
      return {
        ...chunk,
        choices: [{ ...choice!, text: mode === 'thought' ? '' : text, finish_reason: null }],
        ...(mode === 'thought' && { reasoning: text }),
      }
    }
    pending += choice.text
    while (pending) {
      let marker = '<|channel>'
      if (mode === 'header') marker = '\n'
      if (mode === 'thought') marker = '<channel|>'
      const at = pending.indexOf(marker)
      const call = mode === 'text' ? pending.indexOf('<|tool_call>') : -1
      if (call !== -1 && (at === -1 || call < at)) {
        if (call) yield piece(pending.slice(0, call))
        yield piece(pending.slice(call))
        pending = ''
        tool = true
        break
      }
      if (mode === 'header') {
        if (at === -1) break
        if (pending.slice(0, at) !== 'thought')
          throw new ModelError('Gemma returned an unknown thinking channel.')
      } else if (at !== -1) {
        if (at) yield piece(pending.slice(0, at))
      } else {
        const markers = mode === 'text' ? [marker, '<|tool_call>'] : [marker]
        let held = Math.min(pending.length, Math.max(...markers.map((token) => token.length)) - 1)
        while (held && !markers.some((token) => token.startsWith(pending.slice(-held)))) held--
        const text = pending.slice(0, pending.length - held)
        pending = pending.slice(pending.length - held)
        if (text) yield piece(text)
        break
      }
      pending = pending.slice(at + marker.length)
      if (mode === 'text') mode = 'header'
      else if (mode === 'header') mode = 'thought'
      else mode = 'text'
    }
    if (choice.finish_reason) {
      if (mode !== 'text' && choice.finish_reason === 'stop')
        throw new ModelError('Gemma ended inside a thinking block.')
      if (pending && mode !== 'header') yield piece(pending)
      pending = ''
      yield { ...chunk, choices: [{ ...choice, text: '' }] }
    }
  }
}
