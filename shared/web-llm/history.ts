import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import type { ProtocolMessage } from './types.js'

export function lastUserIndex(history: ProtocolMessage[]) {
  for (let index = history.length - 1; index >= 0; index--)
    if (history[index]!.role === 'user') return index
  return -1
}

export function nativeMessages(messages: ProtocolMessage[]): ChatCompletionMessageParam[] {
  return messages.map((message) => {
    const native = { ...message }
    delete native.reasoning
    delete native.response
    delete native.context
    return native
  })
}
