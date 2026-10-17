import { ModelError, type ToolSpec } from '@strands-agents/sdk'
import type { Completion } from '@mlc-ai/web-llm'
import { readArguments } from './value.js'
import { splitThinking } from './thinking.js'
import { stopReason, type StreamChunk, type ToolDelta } from '../chunks.js'

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
    const choice = chunk.choices[0]
    let content = ''
    let calls: ToolDelta[] = []
    if (choice) {
      if (toolText) toolText += choice.text
      else {
        pending += choice.text
        const at = pending.indexOf(begin)
        if (at !== -1) {
          content = pending.slice(0, at)
          toolText = pending.slice(at)
          pending = ''
        } else {
          let held = Math.min(pending.length, begin.length - 1)
          while (held && !begin.startsWith(pending.slice(-held))) held--
          content = pending.slice(0, pending.length - held)
          pending = pending.slice(pending.length - held)
        }
      }
      if (choice.finish_reason) {
        if (toolText && choice.finish_reason === 'stop') calls = readCalls(toolText, tools)
        content += pending
        pending = ''
      }
    }
    yield {
      text: content,
      reasoning: chunk.reasoning ?? '',
      toolText: '',
      calls,
      stop: calls.length ? 'toolUse' : stopReason(choice?.finish_reason),
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
