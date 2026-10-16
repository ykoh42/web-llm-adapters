import { ModelError, type ToolSpec } from '@strands-agents/sdk'
import type { ChatCompletionChunk, Completion } from '@mlc-ai/web-llm'
import { readArguments } from './value.js'
import { splitThinking } from './thinking.js'
import type { StreamChunk } from '../stream.js'

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
    let calls: ChatCompletionChunk.Choice.Delta.ToolCall[] = []
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
      ...chunk,
      object: 'chat.completion.chunk',
      choices: choice
        ? [
            {
              index: 0,
              delta: { ...(content && { content }), ...(calls.length && { tool_calls: calls }) },
              finish_reason: calls.length ? 'tool_calls' : choice.finish_reason,
              ...(choice.logprobs !== undefined && { logprobs: choice.logprobs }),
            },
          ]
        : [],
    }
  }
}

function readCalls(source: string, tools: ToolSpec[]) {
  const calls: ChatCompletionChunk.Choice.Delta.ToolCall[] = []
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
      function: { name, arguments: JSON.stringify(args.input) },
    })
    rest = rest.slice(end.length).trim()
  }
  return calls
}
