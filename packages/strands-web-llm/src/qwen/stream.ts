import { ModelError, type ToolSpec } from '@strands-agents/sdk'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import { stopReason, type StreamChunk, type ToolDelta } from '../stream/chunks.js'
import { textEnd } from '../gemma/markers.js'

const thinkStart = '<think>'
const thinkEnd = '</think>'
const callStart = '<tool_call>'
const callEnd = '</tool_call>'

/** Split Qwen reasoning and parse its tagged function calls across chunks. */
export async function* qwenChunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  onChunk?: (chunk: ChatCompletionChunk) => void,
  parseProtocol = true,
): AsyncGenerator<StreamChunk> {
  let mode: 'text' | 'thinking' | 'tool' = 'text'
  let pending = ''
  let toolJson = ''
  let callIndex = 0
  for await (const chunk of chunks) {
    onChunk?.(chunk)
    const choice = chunk.choices[0]
    if (chunk.choices.length > 1 || (choice && choice.index !== 0))
      throw new ModelError('Strands needs one response per request.')
    pending += choice?.delta.content ?? ''
    let text = ''
    let reasoning = ''
    const calls: ToolDelta[] = []

    while (pending) {
      if (mode === 'tool') {
        const at = pending.indexOf(callEnd)
        if (at === -1) {
          const end = textEnd(pending, [callEnd])
          toolJson += pending.slice(0, end)
          pending = pending.slice(end)
          break
        }
        toolJson += pending.slice(0, at)
        pending = pending.slice(at + callEnd.length)
        calls.push(parseCall(toolJson, tools, callIndex++))
        toolJson = ''
        mode = 'text'
        continue
      }

      if (mode === 'text' && !parseProtocol) {
        text += pending
        pending = ''
        break
      }

      const markers = mode === 'thinking' ? [thinkEnd] : [thinkStart, callStart]
      const at = earliest(pending, markers)
      if (at !== undefined) {
        const value = pending.slice(0, at.index)
        if (mode === 'thinking') reasoning += value
        else text += value
        pending = pending.slice(at.index + at.marker.length)
        mode = mode === 'thinking' ? 'text' : at.marker === thinkStart ? 'thinking' : 'tool'
        continue
      }
      const end = textEnd(pending, markers)
      const value = pending.slice(0, end)
      if (mode === 'thinking') reasoning += value
      else text += value
      pending = pending.slice(end)
      break
    }

    const stop = stopReason(choice?.finish_reason)
    if (stop) {
      if (mode === 'tool') {
        if (stop !== 'maxTokens' && stop !== 'cancelled') {
          const end = textEnd(pending, [callEnd])
          const content = toolJson + pending.slice(0, end)
          calls.push(parseCall(content, tools, callIndex++))
        }
        pending = ''
        toolJson = ''
        mode = 'text'
      }
      if (mode === 'thinking') {
        reasoning += pending
        pending = ''
      } else {
        text += pending
        pending = ''
      }
      mode = 'text'
    }
    yield { text, reasoning, toolText: '', calls, stop, usage: chunk.usage }
  }
}

function earliest(text: string, markers: string[]) {
  let found: { index: number; marker: string } | undefined
  for (const marker of markers) {
    const index = text.indexOf(marker)
    if (index !== -1 && (!found || index < found.index)) found = { index, marker }
  }
  return found
}

function parseCall(text: string, tools: ToolSpec[], index: number): ToolDelta {
  const source = text.trim()
  let value: unknown
  try {
    value = JSON.parse(source)
  } catch (cause) {
    if (!source.endsWith('}')) throw new ModelError('Qwen returned invalid tool JSON.', { cause })
    try {
      value = JSON.parse(source.slice(0, -1))
    } catch {
      throw new ModelError('Qwen returned invalid tool JSON.', { cause })
    }
  }
  if (!isToolCall(value)) throw new ModelError('Qwen returned a malformed tool call.')
  if (!tools.some((tool) => tool.name === value.name))
    throw new ModelError(`Qwen returned unknown tool "${value.name}".`)
  return {
    index,
    id: crypto.randomUUID(),
    name: value.name,
    input: JSON.stringify(value.arguments),
  }
}

function isToolCall(value: unknown): value is { name: string; arguments: Record<string, unknown> } {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const call = value as Record<string, unknown>
  return (
    typeof call.name === 'string' &&
    typeof call.arguments === 'object' &&
    call.arguments !== null &&
    !Array.isArray(call.arguments)
  )
}
