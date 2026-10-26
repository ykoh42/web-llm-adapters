import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'

export function formatUserContent(parts: ChatCompletionContentPart[]) {
  if (parts.every((part) => part.type === 'text')) return parts.map((part) => part.text).join('')

  const content = parts.some((part) => part.type === 'text' && part.text.length === 0)
    ? parts.filter((part) => part.type !== 'text' || part.text.length > 0)
    : parts
  if (content.length === 1) return content

  const first = content.findIndex((part) => part.type === 'text')
  if (first === -1) return content
  let last = content.length - 1
  while (content[last]?.type !== 'text') last--
  if (first === last) return content
  const text = content.slice(first, last + 1)
  if (!text.every((part) => part.type === 'text'))
    throw new Error(
      'WebLLM accepts one text part per user message and cannot preserve text around media blocks.',
    )
  return [
    ...content.slice(0, first),
    { type: 'text' as const, text: text.map((part) => part.text).join('') },
    ...content.slice(last + 1),
  ]
}
