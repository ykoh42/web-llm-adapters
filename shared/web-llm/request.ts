import type { ChatCompletionRequestBase } from '@mlc-ai/web-llm'

export type WebLLMRequest = Omit<
  ChatCompletionRequestBase,
  'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'tool_choice' | 'n' | 'extra_body'
> & {
  extra_body?: NonNullable<ChatCompletionRequestBase['extra_body']> & {
    /** Keep earlier Gemma tool-call reasoning, following its official template. */
    preserve_thinking?: boolean | null
  }
}

const requestFields = [
  'model',
  'messages',
  'prompt',
  'tools',
  'tool_choice',
  'n',
  'stream',
  'stream_options',
]

/** Frameworks own the conversation, tools, and response shape. */
export function checkRequest(request: WebLLMRequest, toolChoice?: 'none') {
  for (const key of requestFields) {
    if (
      key === 'tool_choice' &&
      toolChoice === 'none' &&
      'tool_choice' in request &&
      (request.tool_choice === undefined || request.tool_choice === 'none')
    )
      continue
    if (key in request) throw new Error(`Use the provider API instead of overriding ${key}.`)
  }
}

export function preserveThinking(modelId: string, request: WebLLMRequest) {
  const preserve = request.extra_body?.preserve_thinking
  if (preserve == null) return false
  if (typeof preserve !== 'boolean') throw new Error('preserve_thinking must be a boolean or null.')
  if (!/^gemma-4-/i.test(modelId))
    throw new Error('This WebLLM model does not support preserve_thinking.')
  return preserve
}

/** Strip adapter options and WebLLM's Qwen-only false prefix for Gemma. */
export function nativeRequest<T extends WebLLMRequest>(request: T, modelId: string) {
  const body = request.extra_body
  if (!body) return request
  const omitThinking = body.enable_thinking === false && /^gemma-4-/i.test(modelId)
  if (!('preserve_thinking' in body) && !omitThinking) return request
  const extra_body = { ...body }
  delete extra_body.preserve_thinking
  if (omitThinking) delete extra_body.enable_thinking
  return { ...request, extra_body }
}
