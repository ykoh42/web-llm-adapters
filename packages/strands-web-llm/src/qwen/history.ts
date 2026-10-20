import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import { formatText, formatUser, lastQueryIndex, rejectSignedToolCalls } from '../chat/messages.js'
import { appendQwenTurns, formatQwenAssistant } from '../../../../shared/web-llm/qwen/history.js'
import { formatCall as jsonCall } from '../../../../shared/web-llm/qwen/tools.js'
import { formatCall as xmlCall } from '../../../../shared/web-llm/qwen35/tools.js'

export function qwenHistory(
  messages: Message[],
  options: StreamOptions,
  format: { systemPrompt: string | undefined; keepThoughts: boolean; xml?: boolean },
): ChatCompletionMessageParam[] {
  const output: ChatCompletionMessageParam[] = []
  if (format.systemPrompt !== undefined)
    output.push({ role: 'system', content: format.systemPrompt })
  const lastQuery = lastQueryIndex(messages, options.dynamicTrailingBlocks)
  const formatCall = format.xml ? xmlCall : jsonCall
  let previousWasToolResult = false
  for (const [index, message] of messages.entries()) {
    if (message.role !== 'assistant') {
      previousWasToolResult = appendQwenTurns(
        output,
        formatUser(message.content),
        previousWasToolResult,
      )
      continue
    }

    rejectSignedToolCalls(message.content)
    const thoughts = message.content.filter((block) => block.type === 'reasoningBlock')
    if (
      thoughts.some((block) => block.signature !== undefined || block.redactedContent !== undefined)
    )
      throw new ModelError(
        `${format.xml ? 'Qwen3.5' : 'Qwen'} cannot use signed or redacted reasoning.`,
      )
    const content = message.content.filter((block) => block.type !== 'reasoningBlock')
    const text = formatText(content.filter((block) => block.type !== 'toolUseBlock'))
    const calls = content
      .filter((block) => block.type === 'toolUseBlock')
      .map((call) => formatCall(call.name, call.input))
    const thought =
      format.keepThoughts && index > lastQuery
        ? thoughts.map((block) => block.text ?? '').join('')
        : ''
    output.push({
      role: 'assistant',
      content: formatQwenAssistant(text, thought, calls, format.xml),
    })
    previousWasToolResult = false
  }
  return output
}
