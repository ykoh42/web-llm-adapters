import type { Message } from '@strands-agents/sdk'

export { appendQwenTurns } from '../../../../shared/web-llm/qwen/history.js'

export function lastQwenQuery(messages: Message[]) {
  let last = -1
  for (const [index, message] of messages.entries()) {
    if (
      message.role === 'user' &&
      message.content.some(
        (block) => block.type !== 'toolResultBlock' && block.type !== 'cachePointBlock',
      )
    )
      last = index
  }
  return last
}
