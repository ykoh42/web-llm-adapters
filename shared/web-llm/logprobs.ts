import type { ChatCompletionChunk } from '@mlc-ai/web-llm'

export type WebLLMLogprobs = NonNullable<ChatCompletionChunk['choices'][number]['logprobs']>

export function appendLogprobs(
  current: WebLLMLogprobs | undefined,
  next: WebLLMLogprobs | null | undefined,
): WebLLMLogprobs | undefined {
  if (!next) return current
  if (!current) return { content: next.content ? [...next.content] : null }
  if (next.content) {
    current.content ??= []
    current.content.push(...next.content)
  }
  return current
}
