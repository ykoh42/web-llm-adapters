import { Message, ModelError, type SystemPrompt } from '@strands-agents/sdk'
import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import {
  formatMessages,
  formatAssistant,
  formatUser,
  hasToolContext,
  lastQueryIndex,
} from '../chat/messages.js'

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
  messages = trimThoughts(messages, dynamicTrailingBlocks)
  const last = messages.at(-1)
  const folded = hasToolContext(last, dynamicTrailingBlocks)
  const history: GemmaMessage[] = formatMessages([], systemPrompt)
  for (const message of messages) {
    if (message.role === 'assistant') history.push(assistantTurn(message))
    else history.push(...userTurn(message, Boolean(folded && message === last)))
  }
  return history
}

function assistantTurn(message: Message): GemmaMessage {
  const content = message.content.filter((block) => block.type !== 'reasoningBlock')
  const reasoning = message.content
    .filter((block) => block.type === 'reasoningBlock')
    .map((block) => block.text ?? '')
    .join('')
  return { ...formatAssistant(content), ...(reasoning && { reasoning }) }
}

function userTurn(message: Message, folded: boolean): GemmaMessage[] {
  const turns: GemmaMessage[] = formatUser(message.content, true)
  if (folded) foldReminder(turns)
  return turns
}

function foldReminder(turns: GemmaMessage[]) {
  const reminder = turns.pop()
  const result = turns.at(-1)
  if (reminder?.role !== 'user' || typeof reminder.content !== 'string' || result?.role !== 'tool')
    throw new ModelError('Gemma received invalid trailing context.')
  result.context = reminder.content
}

/** Remove only thoughts that Gemma drops after a new user turn. */
export function trimThoughts(messages: Message[], dynamicTrailingBlocks = 0): Message[] {
  const lastUser = lastQueryIndex(messages, dynamicTrailingBlocks)
  return messages.map((message, index) => {
    if (message.role !== 'assistant') return message
    const thoughts = message.content.filter((block) => block.type === 'reasoningBlock')
    if (
      thoughts.some((block) => block.signature !== undefined || block.redactedContent !== undefined)
    )
      throw new ModelError('Gemma cannot use signed or redacted reasoning.')
    if (index > lastUser || !thoughts.length) return message
    return new Message({
      role: message.role,
      content: message.content.filter((block) => block.type !== 'reasoningBlock'),
    })
  })
}
