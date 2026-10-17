import { ModelError, type ToolSpec } from '@strands-agents/sdk'
import type { Completion } from '@mlc-ai/web-llm'
import { readArguments } from './value.js'
import { splitThinking } from './thinking.js'
import { textEnd } from './markers.js'
import type { StreamChunk, ToolDelta } from '../chunks.js'

const begin = '<|tool_call>'
const end = '<tool_call|>'

export async function* gemmaChunks(
  chunks: AsyncIterable<Completion>,
  tools: ToolSpec[],
  onChunk?: (chunk: Completion) => void,
  prefix = '',
): AsyncGenerator<StreamChunk> {
  let pending = ''
  let toolText = ''
  for await (const chunk of splitThinking(chunks, onChunk, prefix)) {
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
      if (toolText && chunk.stop === 'endTurn') calls = readCalls(toolText, tools)
      content += pending
      pending = ''
    }
    yield {
      text: content,
      reasoning: chunk.reasoning,
      toolText: '',
      calls,
      stop: calls.length ? 'toolUse' : chunk.stop,
      usage: chunk.usage,
    }
  }
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
