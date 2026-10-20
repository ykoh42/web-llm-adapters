import type { ToolCall } from '@earendil-works/pi-ai'
import type { ToolDelta } from '../../../shared/web-llm/stream/chunks.js'
import type { StopReason } from '../../../shared/web-llm/types.js'
import { parseJSON } from '../../../shared/web-llm/json.js'
import { checkToolChoice } from '../../../shared/web-llm/stream/choice.js'

export function toolCalls() {
  const calls = new Map<number, { id?: string; name: string; input: string }>()

  function append(deltas: ToolDelta[]) {
    for (const delta of deltas) {
      const call = calls.get(delta.index) ?? { name: '', input: '' }
      if (delta.id) {
        if (call.id && call.id !== delta.id) throw new Error('WebLLM changed a tool call ID.')
        call.id = delta.id
      }
      call.name += delta.name ?? ''
      call.input += delta.input ?? ''
      calls.set(delta.index, call)
    }
  }

  function finish(stop: StopReason, choice: 'none' | undefined, toolNames: string[]) {
    const ordered = [...calls.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, call]) => call)
    checkToolChoice(
      ordered.map((call) => call.name),
      choice,
      stop,
    )
    if (stop === 'maxTokens') return []
    const ids = new Set<string>()
    const complete = ordered.map((call) => {
      if (!toolNames.includes(call.name))
        throw new Error(`WebLLM returned an unknown tool: ${call.name}.`)
      const input = parseJSON(call.input)
      if (input === null || typeof input !== 'object' || Array.isArray(input))
        throw new Error('WebLLM tool arguments must be a JSON object.')
      const id = call.id ?? crypto.randomUUID()
      if (ids.has(id)) throw new Error('WebLLM returned duplicate tool call IDs.')
      ids.add(id)
      return {
        call: { type: 'toolCall', id, name: call.name, arguments: input } satisfies ToolCall,
        delta: call.input,
      }
    })
    if (stop === 'toolUse' && !complete.length) throw new Error('WebLLM returned no tool calls.')
    return complete
  }

  return { append, finish }
}
