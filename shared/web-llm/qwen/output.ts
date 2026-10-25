import type { StopReason } from '../types.js'
import type { StreamChunk } from '../stream/chunks.js'

type Part = { type: 'text' | 'reasoning' | 'tool'; value: string }

/** A tool marker keeps output ordered without publishing protocol syntax as text. */
export class QwenOutput {
  private readonly parts: Part[] = []

  add(type: Part['type'], value: string) {
    if (!value) return
    const content = type === 'tool' ? '' : value
    const previous = this.parts.at(-1)
    if (previous?.type === type) previous.value += content
    else this.parts.push({ type, value: content })
  }

  *read(end?: StopReason | 'incomplete'): Generator<StreamChunk> {
    let consumed = 0
    for (const part of this.parts) {
      if (part.type === 'tool' && !end) break
      consumed++
      if (part.type === 'tool') continue
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
