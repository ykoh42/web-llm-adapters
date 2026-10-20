import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'

/** Replay thinking with the separators used by Qwen's chat templates. */
export function formatQwenAssistant(
  text: string,
  thought: string,
  calls: readonly string[],
  xml = false,
) {
  const reasoning = xml ? thought.trim() : thought.replace(/^\n+|\n+$/g, '')
  let answer = text
  if (reasoning) answer = xml ? text.trim() : text.replace(/^\n+/, '')
  const prefix = reasoning ? `<think>\n${reasoning}\n</think>\n\n` : ''
  const separator = xml ? '\n\n' : '\n'
  const gap = answer && calls.length ? separator : ''
  return prefix + answer + gap + calls.join('\n')
}

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
