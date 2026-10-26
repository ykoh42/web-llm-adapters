import type { ChatCompletionChunk } from '@mlc-ai/web-llm'

export type WebLLMLogprobs = NonNullable<ChatCompletionChunk['choices'][number]['logprobs']>

export function appendLogprobs(
  current: WebLLMLogprobs | undefined,
  next: WebLLMLogprobs | null | undefined,
): WebLLMLogprobs | undefined {
  if (!next) return current
  const content = next.content ? structuredClone(next.content) : next.content
  if (!current) return { content }
  if (content) {
    current.content ??= []
    current.content.push(...content)
  }
  return current
}
