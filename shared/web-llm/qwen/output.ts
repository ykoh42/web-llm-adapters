import type { StopReason } from '../types.js'
import type { StreamChunk } from '../stream/chunks.js'

type Part = { type: 'text' | 'reasoning' | 'tool'; value: string }

/** Keep tool syntax buffered until the stream ends. */
export class QwenOutput {
  private readonly parts: Part[] = []

  add(type: Part['type'], value: string) {
    if (!value) return
    const previous = this.parts.at(-1)
    if (previous?.type === type) previous.value += value
    else this.parts.push({ type, value })
  }

  *read(end?: StopReason | 'incomplete'): Generator<StreamChunk> {
    const truncated = end === 'maxTokens' || end === 'cancelled' || end === 'incomplete'
    let consumed = 0
    for (const part of this.parts) {
      if (part.type === 'tool' && !end) break
      consumed++
      if (part.type === 'tool' && !truncated) continue
      yield {
        text: part.type === 'reasoning' ? '' : part.value,
        reasoning: part.type === 'reasoning' ? part.value : '',
        calls: [],
        stop: undefined,
        usage: undefined,
      }
    }
    this.parts.splice(0, consumed)
  }
}
