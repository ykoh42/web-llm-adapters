import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import type { GemmaMessage } from '../../../../shared/web-llm/gemma/prompt.js'
import type { ToolSpec, ToolChoice } from '../../../../shared/web-llm/types.js'
import * as qwen from '../../../../shared/web-llm/qwen/tools.js'
import * as qwen35 from '../../../../shared/web-llm/qwen35/tools.js'
import { appendQwenTurns, formatQwenAssistant } from '../../../../shared/web-llm/qwen/history.js'

export function trimThoughts(history: GemmaMessage[]) {
  const last = lastUser(history)
  return history.map((message, index) => (index > last ? message : withoutReasoning(message)))
}

export function qwenHistory(
  modelId: string,
  history: GemmaMessage[],
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
  const last = lastUser(history)
  let previousWasTool = false
  for (const [index, message] of history.entries()) {
    if (message.role === 'system') continue
    if (message.role !== 'assistant') {
      previousWasTool = appendQwenTurns(output, [message], previousWasTool)
      continue
    }
    const thought = !modelId.startsWith('Qwen2.5-') && index > last ? (message.reasoning ?? '') : ''
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

function lastUser(history: GemmaMessage[]) {
  let last = -1
  for (const [index, message] of history.entries()) if (message.role === 'user') last = index
  return last
}

function withoutReasoning(message: GemmaMessage): GemmaMessage {
  const copy = { ...message }
  delete copy.reasoning
  return copy
}
