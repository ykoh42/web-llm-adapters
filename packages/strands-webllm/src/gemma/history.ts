import { Message, ModelError, type SystemPrompt } from '@strands-agents/sdk'
import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import { formatMessages } from '../messages.js'

type GemmaMessage = ChatCompletionMessageParam & {
  reasoning?: string
  response?: unknown
  context?: string
}

/** Keep thoughts in the current tool loop, without changing the Agent's history. */
export function gemmaHistory(
  messages: Message[],
  systemPrompt?: SystemPrompt,
  dynamicTrailingBlocks = 0,
): GemmaMessage[] {
  const last = messages.at(-1)
  const folded =
    last?.role === 'user' &&
    dynamicTrailingBlocks > 0 &&
    last.content.slice(-dynamicTrailingBlocks).some((block) => block.type === 'textBlock') &&
    last.content
      .slice(0, -dynamicTrailingBlocks)
      .every((block) => block.type === 'toolResultBlock' || block.type === 'cachePointBlock') &&
    last.content.some((block) => block.type === 'toolResultBlock')
  let lastUser = -1
  for (const [index, message] of messages.entries()) {
    if (
      message.role === 'user' &&
      !(folded && message === last) &&
      message.content.some(
        (block) => block.type !== 'toolResultBlock' && block.type !== 'cachePointBlock',
      )
    )
      lastUser = index
  }
  const history: GemmaMessage[] = formatMessages([], systemPrompt)
  for (const [index, message] of messages.entries()) {
    if (message.role !== 'assistant') {
      // WebLLM has no provider prompt cache to protect. Put ephemeral context before
      // the user's request so the request remains the last text Gemma reads.
      const ordered =
        message === last && message.role === 'user' && dynamicTrailingBlocks > 0 && !folded
          ? new Message({
              role: 'user',
              content: [
                ...message.content.slice(-dynamicTrailingBlocks),
                ...message.content.slice(0, -dynamicTrailingBlocks),
              ],
            })
          : message
      const formatted: GemmaMessage[] = formatMessages([ordered])
      let resultIndex = 0
      for (const block of message.content) {
        if (block.type !== 'toolResultBlock') continue
        const result = formatted[resultIndex++]!
        if (block.status === 'error') continue
        const first = block.content[0]
        if (block.content.length === 1 && first?.type === 'jsonBlock') result.response = first.json
        else if (block.content.length > 1)
          result.response = block.content.map((item) => item.toJSON())
      }
      if (folded && message === last) {
        // Harness reminders belong to this tool response, not a new user turn.
        const reminder = formatted.pop()
        const result = formatted.at(-1)
        if (
          reminder?.role !== 'user' ||
          typeof reminder.content !== 'string' ||
          result?.role !== 'tool'
        )
          throw new ModelError('Gemma received invalid trailing context.')
        result.context = reminder.content
      }
      history.push(...formatted)
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
