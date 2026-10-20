import { ModelError, type ToolSpec } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import {
  singleChoice,
  stopReason,
  type StreamChunk,
  type ToolDelta,
  type ChunkOptions,
} from '../stream/chunks.js'
import { textEnd } from '../gemma/markers.js'
import { readValue } from './value.js'
import { QwenOutput } from '../qwen/output.js'

const thinkStart = '<think>'
const thinkEnd = '</think>'
const callStart = '<tool_call>'
const callEnd = '</tool_call>'

/** Split Qwen3.5 reasoning and parse its XML function and parameter tags. */
export async function* qwen35Chunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  {
    onChunk,
    signal,
    parseProtocol = true,
  }: ChunkOptions<ChatCompletionChunk> & {
    parseProtocol?: boolean
  } = {},
): AsyncGenerator<StreamChunk> {
  const textMarkers = tools.length ? [thinkStart, callStart] : [thinkStart]
  let mode: 'text' | 'thinking' | 'tool' = 'text'
  let pending = ''
  let toolText = ''
  const output = new QwenOutput()
  let callIndex = 0
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of chunks) {
      if (signal?.aborted) return
      onChunk?.(chunk)
      if (signal?.aborted) return
      const choice = singleChoice(chunk.choices)
      pending += choice?.delta.content ?? ''
      const calls: ToolDelta[] = []

      while (pending) {
        if (mode === 'tool') {
          const at = pending.indexOf(callEnd)
          if (at === -1) {
            const end = textEnd(pending, [callEnd])
            output.add('tool', pending.slice(0, end))
            toolText += pending.slice(0, end)
            pending = pending.slice(end)
            break
          }
          toolText += pending.slice(0, at)
          output.add('tool', pending.slice(0, at) + callEnd)
          pending = pending.slice(at + callEnd.length)
          const parsed = parseCalls(toolText, tools, callIndex)
          calls.push(...parsed)
          callIndex += parsed.length
          toolText = ''
          mode = 'text'
          continue
        }

        if (mode === 'text' && !parseProtocol) {
          output.add('text', pending)
          pending = ''
          break
        }

        const markers = mode === 'thinking' ? [thinkEnd] : textMarkers
        const at = earliest(pending, markers)
        if (at !== undefined) {
          const value = pending.slice(0, at.index)
          if (mode === 'thinking') output.add('reasoning', value)
          else output.add('text', value)
          pending = pending.slice(at.index + at.marker.length)
          if (mode === 'thinking') mode = 'text'
          else if (at.marker === thinkStart) mode = 'thinking'
          else {
            output.add('tool', callStart)
            mode = 'tool'
          }
          continue
        }
        const end = textEnd(pending, markers)
        const value = pending.slice(0, end)
        if (mode === 'thinking') output.add('reasoning', value)
        else output.add('text', value)
        pending = pending.slice(end)
        break
      }

      const stop = stopReason(choice?.finish_reason)
      if (stop) {
        if (mode === 'tool' && stop !== 'maxTokens' && stop !== 'cancelled')
          throw new ModelError('Qwen3.5 ended inside a tool call.')
        if (mode === 'tool') {
          output.add('tool', pending)
          pending = ''
          toolText = ''
        }
        if (mode === 'thinking') {
          output.add('reasoning', pending)
          pending = ''
        } else {
          output.add('text', pending)
          pending = ''
        }
        mode = 'text'
      }
      yield* output.read(stop)
      if (calls.length || stop || chunk.usage)
        yield { text: '', reasoning: '', calls, stop, usage: chunk.usage }
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  output.add(mode === 'thinking' ? 'reasoning' : 'text', pending)
  yield* output.read('incomplete')
  if (failure && !signal?.aborted) throw failure.error
}

function earliest(text: string, markers: string[]) {
  let found: { index: number; marker: string } | undefined
  for (const marker of markers) {
    const index = text.indexOf(marker)
    if (index !== -1 && (!found || index < found.index)) found = { index, marker }
  }
  return found
}

function parseCalls(source: string, tools: ToolSpec[], index: number): ToolDelta[] {
  const calls: ToolDelta[] = []
  const functions = /<function=([^>\s]+)>([\s\S]*?)<\/function>/g
  for (const match of source.matchAll(functions)) {
    const name = match[1]!
    const tool = tools.find((item) => item.name === name)
    if (!tool) throw new ModelError(`Qwen3.5 returned unknown tool "${name}".`)
    const input = Object.create(null) as Record<string, unknown>
    const body = match[2]!
    const parameters = /<parameter=([^>]*)>([\s\S]*?)(?:<\/parameter>|(?=<parameter=))/g
    const values = [...body.matchAll(parameters)]
    // Like vLLM's Qwen parser, ignore redundant closers outside complete parameters.
    if (body.replace(parameters, '').replaceAll('</parameter>', '').trim())
      throw new ModelError('Qwen3.5 returned a malformed tool parameter.')
    for (const parameter of values) {
      const key = parameter[1]!
      if (Object.hasOwn(input, key)) throw new ModelError(`Qwen3.5 repeated parameter "${key}".`)
      input[key] = readValue(parameter[2]!, tool, key)
    }
    calls.push({
      index: index + calls.length,
      id: crypto.randomUUID(),
      name,
      input: JSON.stringify(input),
    })
  }
  if (!calls.length || source.replace(functions, '').trim())
    throw new ModelError('Qwen3.5 returned a malformed tool call.')
  return calls
}
