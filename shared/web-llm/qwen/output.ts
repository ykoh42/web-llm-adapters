import type { StopReason } from '../types.js'
import { noToolCalls, type StreamChunk, type ToolDelta } from '../stream/chunks.js'

type Part = { type: 'text' | 'reasoning'; value: string } | { type: 'tool'; calls?: ToolDelta[] }

/** A tool marker keeps output ordered without publishing protocol syntax as text. */
export class QwenOutput {
  private readonly parts: Part[] = []

  add(type: Part['type'], value: string) {
    if (!value) return
    if (type === 'tool') {
      this.parts.push({ type })
      return
    }
    const previous = this.parts.at(-1)
    if (previous?.type === type) previous.value += value
    else this.parts.push({ type, value })
  }

  finishTool(calls: ToolDelta[]) {
    const part = this.parts.at(-1)
    if (part?.type !== 'tool') throw new Error('Missing Qwen tool marker.')
    part.calls = calls
  }

  *read(end?: StopReason | 'incomplete'): Generator<StreamChunk> {
    let consumed = 0
    for (const part of this.parts) {
      if (part.type === 'tool' && !part.calls && !end) break
      consumed++
      if (part.type === 'tool') {
        if (part.calls?.length && end !== 'incomplete')
          yield { text: '', reasoning: '', calls: part.calls, stop: undefined, usage: undefined }
        continue
      }
      yield {
        text: part.type === 'reasoning' ? '' : part.value,
        reasoning: part.type === 'reasoning' ? part.value : '',
        calls: noToolCalls,
        stop: undefined,
        usage: undefined,
      }
    }
    this.parts.splice(0, consumed)
  }
}
