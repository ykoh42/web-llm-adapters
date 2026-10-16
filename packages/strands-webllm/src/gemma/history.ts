import { Message, ModelError, type SystemPrompt } from '@strands-agents/sdk'
import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import { formatMessages } from '../messages.js'

type GemmaMessage = ChatCompletionMessageParam & { reasoning?: string }

/** Keep thoughts in the current tool loop, without changing the Agent's history. */
export function gemmaHistory(messages: Message[], systemPrompt?: SystemPrompt): GemmaMessage[] {
  let lastUser = -1
  for (const [index, message] of messages.entries()) {
    if (
      message.role === 'user' &&
      message.content.some(
        (block) => block.type !== 'toolResultBlock' && block.type !== 'cachePointBlock',
      )
    )
      lastUser = index
  }
  const history: GemmaMessage[] = formatMessages([], systemPrompt)
  for (const [index, message] of messages.entries()) {
    if (message.role !== 'assistant') {
      history.push(...formatMessages([message]))
      continue
    }
    const thoughts = message.content.filter((block) => block.type === 'reasoningBlock')
    if (
      thoughts.some((block) => block.signature !== undefined || block.redactedContent !== undefined)
    )
      throw new ModelError('Gemma cannot use signed or redacted reasoning.')
    const clean = new Message({
      role: 'assistant',
      content: message.content.filter((block) => block.type !== 'reasoningBlock'),
    })
    const reasoning = thoughts.map((block) => block.text ?? '').join('\n')
    history.push({
      ...formatMessages([clean])[0]!,
      ...(index > lastUser && reasoning && { reasoning }),
    })
  }
  return history
}
