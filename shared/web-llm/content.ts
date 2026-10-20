import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'

export function formatUserContent(parts: ChatCompletionContentPart[]) {
  if (parts.every((part) => part.type === 'text')) return parts.map((part) => part.text).join('')

  const output: ChatCompletionContentPart[] = []
  const textParts: string[] = []
  let textRuns = 0
  let inText = false
  let firstTextIndex = -1
  for (const part of parts) {
    if (part.type !== 'text') {
      inText = false
      output.push(part)
      continue
    }

    if (!inText) textRuns++
    inText = true
    if (firstTextIndex === -1) firstTextIndex = output.length
    textParts.push(part.text)
    output.push(part)
  }

  if (textRuns > 1)
    throw new Error(
      'WebLLM accepts one text part per user message and cannot preserve text around media blocks.',
    )
  if (textParts.length < 2) return parts

  output.splice(firstTextIndex, textParts.length, { type: 'text', text: textParts.join('') })
  return output
}
