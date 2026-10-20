import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'

export function formatUserContent(parts: ChatCompletionContentPart[]) {
  if (parts.every((part) => part.type === 'text')) return parts.map((part) => part.text).join('')

  const first = parts.findIndex((part) => part.type === 'text')
  if (first === -1) return parts
  let last = parts.length - 1
  while (parts[last]?.type !== 'text') last--
  if (first === last) return parts
  const text = parts.slice(first, last + 1)
  if (!text.every((part) => part.type === 'text'))
    throw new Error(
      'WebLLM accepts one text part per user message and cannot preserve text around media blocks.',
    )
  return [
    ...parts.slice(0, first),
    { type: 'text' as const, text: text.map((part) => part.text).join('') },
    ...parts.slice(last + 1),
  ]
}
