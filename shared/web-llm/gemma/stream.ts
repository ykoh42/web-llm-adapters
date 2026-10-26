import { ModelError, type ToolSpec } from '../types.js'
import type { Completion } from '@mlc-ai/web-llm'
import { readArguments } from './value.js'
import { splitThinking, type GemmaStreamOptions } from './thinking.js'
import { toolStart } from './markers.js'
import { textEnd } from '../stream/markers.js'
import { noToolCalls, type StreamChunk, type ToolDelta } from '../stream/chunks.js'

const end = '<tool_call|>'
const toolStartMarkers = [toolStart]

export async function* gemmaChunks(
  chunks: AsyncIterable<Completion>,
  tools: ToolSpec[],
  options: GemmaStreamOptions = {},
): AsyncGenerator<StreamChunk> {
  const { signal } = options
  const parseTools = tools.length > 0
  let pending = ''
  let toolText = ''
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of splitThinking(chunks, { ...options, parseTools })) {
      if (signal?.aborted) return
      if (chunk.type === 'reasoning') {
        if (pending) {
          yield {
            text: pending,
            reasoning: '',
            calls: noToolCalls,
            stop: undefined,
            usage: undefined,
          }
          pending = ''
        }
        if (chunk.text || chunk.stop || chunk.usage)
          yield {
            text: '',
            reasoning: chunk.text,
            calls: noToolCalls,
            stop: chunk.stop,
            usage: chunk.usage,
          }
        continue
      }
      let content = ''
      let calls: ToolDelta[] | undefined
      if (!parseTools) content = chunk.text
      else if (toolText) toolText += chunk.text
      else {
        pending += chunk.text
        const at = pending.indexOf(toolStart)
        if (at !== -1) {
          content = pending.slice(0, at)
          toolText = pending.slice(at)
          pending = ''
        } else {
          const end = textEnd(pending, toolStartMarkers)
          content = pending.slice(0, end)
          pending = pending.slice(end)
        }
      }
      if (chunk.stop) {
        if (toolText) {
          if (chunk.stop === 'endTurn') calls = readCalls(toolText, tools)
          else content += toolText
          toolText = ''
        }
        content += pending
        pending = ''
      }
      yield {
        text: content,
        reasoning: '',
        calls: calls ?? noToolCalls,
        stop: calls?.length ? 'toolUse' : chunk.stop,
        usage: chunk.usage,
      }
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  if (toolText || pending)
    yield {
      text: toolText + pending,
      reasoning: '',
      calls: noToolCalls,
      stop: undefined,
      usage: undefined,
    }
  if (failure && !signal?.aborted) throw failure.error
}

function readCalls(source: string, tools: ToolSpec[]) {
  const calls: ToolDelta[] = []
  const names = tools.map((tool) => tool.name).sort((left, right) => right.length - left.length)
  let rest = source.trim()
  while (rest) {
    const name = names.find((name) => {
      const prefix = `${toolStart}${name}`
      return rest.startsWith(prefix) && /^\s*\{/.test(rest.slice(prefix.length))
    })
    if (name === undefined)
      throw new ModelError('Gemma returned an unknown or malformed tool call.')
    const prefix = `${toolStart}${name}`
    const args = readArguments(rest.slice(prefix.length))
    rest = rest.slice(prefix.length + args.end).trimStart()
    if (!rest.startsWith(end)) throw new ModelError('Gemma returned an incomplete tool call.')
    calls.push({
      index: calls.length,
      id: crypto.randomUUID(),
      name,
      input: JSON.stringify(args.input),
    })
    rest = rest.slice(end.length).trim()
  }
  return calls
}
