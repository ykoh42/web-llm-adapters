import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import type { Message } from '@strands-agents/sdk'

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
