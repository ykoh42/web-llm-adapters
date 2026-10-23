import type { ProtocolMessage } from '../types.js'
import { lastUserIndex } from '../history.js'

export function trimThoughts(history: ProtocolMessage[], preserve = false) {
  const last = lastUserIndex(history)
  return history.map((message, index) =>
    index > last || (preserve && message.role === 'assistant' && message.tool_calls?.length)
      ? message
      : withoutReasoning(message),
  )
}

function withoutReasoning(message: ProtocolMessage): ProtocolMessage {
  if (!Object.hasOwn(message, 'reasoning')) return message
  const copy = { ...message }
  delete copy.reasoning
  return copy
}
