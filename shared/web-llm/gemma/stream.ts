import { ModelError, type ToolSpec } from '../types.js'
import type { Completion } from '@mlc-ai/web-llm'
import { readArguments } from './value.js'
import { splitThinking, type GemmaStreamOptions } from './thinking.js'
import { textEnd } from './markers.js'
import type { StreamChunk, ToolDelta } from '../stream/chunks.js'

const begin = '<|tool_call>'
const end = '<tool_call|>'

export async function* gemmaChunks(
  chunks: AsyncIterable<Completion>,
  tools: ToolSpec[],
  options: GemmaStreamOptions = {},
): AsyncGenerator<StreamChunk> {
  const { signal } = options
  let pending = ''
  let toolText = ''
  let failure: { error: unknown } | undefined
  try {
    for await (const chunk of splitThinking(chunks, options)) {
      if (signal?.aborted) return
      if (chunk.type === 'reasoning') {
        if (pending) {
          yield { text: pending, reasoning: '', calls: [], stop: undefined, usage: undefined }
          pending = ''
        }
        if (chunk.text || chunk.stop || chunk.usage)
          yield { text: '', reasoning: chunk.text, calls: [], stop: chunk.stop, usage: chunk.usage }
        continue
      }
      let content = ''
      let calls: ToolDelta[] = []
      if (toolText) toolText += chunk.text
      else {
        pending += chunk.text
        const at = pending.indexOf(begin)
        if (at !== -1) {
          content = pending.slice(0, at)
          toolText = pending.slice(at)
          pending = ''
        } else {
          const end = textEnd(pending, [begin])
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
        calls,
        stop: calls.length ? 'toolUse' : chunk.stop,
        usage: chunk.usage,
      }
    }
  } catch (error) {
    failure = { error }
  }
  if (signal?.aborted) return
  if (toolText || pending)
    yield { text: toolText + pending, reasoning: '', calls: [], stop: undefined, usage: undefined }
  if (failure && !signal?.aborted) throw failure.error
}

function readCalls(source: string, tools: ToolSpec[]) {
  const calls: ToolDelta[] = []
  let rest = source.trim()
  while (rest) {
    const head = /^<\|tool_call>call:([^\s{]+)\s*/.exec(rest)
    if (!head || !tools.some((tool) => tool.name === head[1]))
      throw new ModelError('Gemma returned an unknown or malformed tool call.')
    const name = head[1]!
    const args = readArguments(rest.slice(head[0].length))
    rest = rest.slice(head[0].length + args.end).trimStart()
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
