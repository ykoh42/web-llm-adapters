import type { ToolCall } from '@earendil-works/pi-ai'
import type { ToolDelta } from '../../../shared/web-llm/stream/chunks.js'
import type { StopReason, ToolChoice } from '../../../shared/web-llm/types.js'
import { parseJSON } from '../../../shared/web-llm/json.js'
import { checkToolCallIds, checkToolChoice } from '../../../shared/web-llm/stream/choice.js'

export interface ToolPolicy {
  choice: ToolChoice | 'none' | undefined
  toolNames: readonly string[]
}

export interface ToolCallUpdate {
  index: number
  id: string | undefined
  name: string
  argumentsDelta: string
  argumentsText: string
}

export function toolCalls() {
  const calls = new Map<number, { id?: string; name: string; input: string }>()

  function append(deltas: readonly ToolDelta[]) {
    const updates: ToolCallUpdate[] = []
    for (const delta of deltas) {
      const call = calls.get(delta.index) ?? { name: '', input: '' }
      if (delta.id) {
        if (call.id && call.id !== delta.id) throw new Error('WebLLM changed a tool call ID.')
        call.id = delta.id
      }
      call.name += delta.name ?? ''
      const argumentsDelta = delta.input ?? ''
      call.input += argumentsDelta
      calls.set(delta.index, call)
      updates.push({
        index: delta.index,
        id: call.id,
        name: call.name,
        argumentsDelta,
        argumentsText: call.input,
      })
    }
    return updates
  }

  function finish(stop: StopReason, { choice, toolNames }: ToolPolicy) {
    const ordered = [...calls.entries()].sort(([left], [right]) => left - right)
    const orderedCalls = ordered.map(([, call]) => call)
    checkToolChoice(orderedCalls, choice, stop, toolNames)
    if (stop === 'maxTokens') return []
    checkToolCallIds(orderedCalls)
    const complete = ordered.map(([index, call]) => {
      if (!toolNames.includes(call.name))
        throw new Error(`WebLLM returned an unknown tool: ${call.name}.`)
      const input = parseJSON(call.input)
      if (input === null || typeof input !== 'object' || Array.isArray(input))
        throw new Error('WebLLM tool arguments must be a JSON object.')
      const id = call.id ?? crypto.randomUUID()
      return {
        index,
        call: { type: 'toolCall', id, name: call.name, arguments: input } satisfies ToolCall,
        delta: call.input,
      }
    })
    if (stop === 'toolUse' && !complete.length) throw new Error('WebLLM returned no tool calls.')
    return complete
  }

  return { append, finish }
}
