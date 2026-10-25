import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import type { ProtocolMessage, ToolSpec, ToolChoice } from '../types.js'
import { lastUserIndex } from '../history.js'
import * as qwen from './tools.js'
import * as qwen35 from '../qwen35/tools.js'

export function qwenHistory(
  modelId: string,
  history: ProtocolMessage[],
  tools: ToolSpec[],
  choice: ToolChoice | undefined,
  xml: boolean,
): ChatCompletionMessageParam[] {
  const output: ChatCompletionMessageParam[] = []
  const system = history[0]?.role === 'system' ? history[0].content : undefined
  const declarations = (xml ? qwen35 : qwen).toolPrompt(tools, choice)
  const parts = xml
    ? [declarations, system]
    : [
        system ??
          (modelId.startsWith('Qwen2.5-')
            ? 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
            : ''),
        declarations,
      ]
  const prompt = parts.filter((part) => part !== undefined && part !== '').join('\n\n')
  if (prompt || system === '') output.push({ role: 'system', content: prompt })
  const last = lastUserIndex(history)
  let previousWasTool = false
  for (const [index, message] of history.entries()) {
    if (message.role === 'system') continue
    if (message.role !== 'assistant') {
      previousWasTool = appendQwenTurns(output, [message], previousWasTool)
      continue
    }
    const thought =
      !modelId.startsWith('Qwen2.5-') && index > last ? (message.reasoning ?? '') : undefined
    const calls = (message.tool_calls ?? []).map((call) => {
      const input: unknown = JSON.parse(call.function.arguments)
      return (xml ? qwen35 : qwen).formatCall(call.function.name, input)
    })
    output.push({
      role: 'assistant',
      content: formatQwenAssistant(message.content ?? '', thought, calls, xml),
      ...(message.name !== undefined && message.name !== 'model' && { name: message.name }),
    })
    previousWasTool = false
  }
  return output
}

/** Replay thinking with the separators used by Qwen's chat templates. */
export function formatQwenAssistant(
  text: string,
  thought: string | undefined,
  calls: readonly string[],
  xml = false,
) {
  const reasoning = xml ? (thought ?? '').trim() : trimNewlines(thought ?? '')
  let answer = text
  if (reasoning) answer = xml ? text.trim() : text.replace(/^\n+/, '')
  const prefix =
    reasoning || (xml && thought !== undefined) ? `<think>\n${reasoning}\n</think>\n\n` : ''
  const separator = xml ? '\n\n' : '\n'
  const gap = answer && calls.length ? separator : ''
  return prefix + answer + gap + calls.join('\n')
}

function trimNewlines(text: string) {
  let start = 0
  let end = text.length
  while (text[start] === '\n') start++
  while (end > start && text[end - 1] === '\n') end--
  return text.slice(start, end)
}

/** Keep adjacent tool replies in one user turn, as Qwen's templates do. */
export function appendQwenTurns(
  output: ChatCompletionMessageParam[],
  turns: ChatCompletionMessageParam[],
  previousWasToolResult: boolean,
) {
  for (const turn of turns) {
    if (turn.role !== 'tool') {
      output.push(turn)
      previousWasToolResult = false
      continue
    }

    const content = turn.content
    const response = `<tool_response>\n${content}\n</tool_response>`
    const previous = output.at(-1)
    if (previousWasToolResult && previous?.role === 'user' && typeof previous.content === 'string')
      previous.content += `\n${response}`
    else output.push({ role: 'user', content: response })
    previousWasToolResult = true
  }
  return previousWasToolResult
}
