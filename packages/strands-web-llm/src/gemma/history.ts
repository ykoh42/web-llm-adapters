import { Message, ModelError, type SystemPrompt } from '@strands-agents/sdk'
import type { ProtocolMessage } from '../../../../shared/web-llm/types.js'
import {
  formatMessages,
  formatAssistant,
  formatUser,
  hasToolContext,
  lastQueryIndex,
} from '../chat/messages.js'

/** Keep thoughts in the current tool loop, without changing the Agent's history. */
export function gemmaHistory(
  messages: Message[],
  systemPrompt?: SystemPrompt,
  dynamicTrailingBlocks = 0,
  preserve = false,
): ProtocolMessage[] {
  messages = trimThoughts(messages, dynamicTrailingBlocks, preserve)
  const last = messages.at(-1)
  const folded = hasToolContext(last, dynamicTrailingBlocks)
  const history: ProtocolMessage[] = formatMessages([], systemPrompt)
  for (const message of messages) {
    if (message.role === 'assistant') history.push(assistantTurn(message))
    else history.push(...userTurn(message, Boolean(folded && message === last)))
  }
  return history
}

function assistantTurn(message: Message): ProtocolMessage {
  const content = message.content.filter((block) => block.type !== 'reasoningBlock')
  const reasoning = message.content
    .filter((block) => block.type === 'reasoningBlock')
    .map((block) => block.text ?? '')
    .join('')
  return { ...formatAssistant(content), ...(reasoning && { reasoning }) }
}

function userTurn(message: Message, folded: boolean): ProtocolMessage[] {
  const turns: ProtocolMessage[] = formatUser(message.content, true)
  if (folded) foldReminder(turns)
  return turns
}

function foldReminder(turns: ProtocolMessage[]) {
  const reminder = turns.pop()
  const result = turns.at(-1)
  if (reminder?.role !== 'user' || typeof reminder.content !== 'string' || result?.role !== 'tool')
    throw new ModelError('Gemma received invalid trailing context.')
  result.context = reminder.content
}

/** Remove only thoughts that Gemma drops after a new user turn. */
export function trimThoughts(
  messages: Message[],
  dynamicTrailingBlocks = 0,
  preserve = false,
): Message[] {
  const lastUser = lastQueryIndex(messages, dynamicTrailingBlocks)
  return messages.map((message, index) => {
    if (message.role !== 'assistant') return message
    let hasThoughts = false
    for (const block of message.content) {
      if (block.type !== 'reasoningBlock') continue
      if (block.signature !== undefined || block.redactedContent !== undefined)
        throw new ModelError('Gemma cannot use signed or redacted reasoning.')
      hasThoughts = true
    }
    if (
      index > lastUser ||
      !hasThoughts ||
      (preserve && message.content.some((block) => block.type === 'toolUseBlock'))
    )
      return message
    return new Message({
      ...message,
      content: message.content.filter((block) => block.type !== 'reasoningBlock'),
    })
  })
}
