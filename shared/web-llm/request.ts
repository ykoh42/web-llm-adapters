import type { ChatCompletionRequestBase } from '@mlc-ai/web-llm'

export type WebLLMRequest = Omit<
  ChatCompletionRequestBase,
  'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'tool_choice' | 'n' | 'extra_body'
> & {
  /** Keep WebLLM's optional usage chunk enabled by default; callers may disable it. */
  stream_options?: ChatCompletionRequestBase['stream_options']
  /** Constrain verified Qwen tool formats with WebLLM 0.2.86 or later. */
  constrainedToolCalls?: boolean
  extra_body?: NonNullable<ChatCompletionRequestBase['extra_body']> & {
    /** Keep earlier Gemma tool-call reasoning, following its official template. */
    preserve_thinking?: boolean | null
  }
}

const requestFields = ['model', 'messages', 'prompt', 'tools', 'tool_choice', 'n', 'stream']

export function streamOptions(request: WebLLMRequest) {
  return request.stream_options === undefined ? { include_usage: true } : request.stream_options
}

/** Keep preserve_thinking when a higher-priority request replaces extra_body. */
export function preserveThinkingRequest(request: WebLLMRequest, ...bodies: unknown[]) {
  for (const body of bodies) {
    const preserve = preserveThinkingOption(body)
    if (preserve === undefined) continue
    request.extra_body = { ...request.extra_body, preserve_thinking: preserve }
    return
  }
}

function preserveThinkingOption(body: unknown) {
  if (typeof body !== 'object' || body === null || !Object.hasOwn(body, 'preserve_thinking'))
    return undefined
  const value: unknown = Reflect.get(body, 'preserve_thinking')
  if (value !== undefined && value !== null && typeof value !== 'boolean')
    throw new Error('preserve_thinking must be a boolean or null.')
  return value
}

/** Frameworks own the conversation, tools, and response shape. */
export function checkRequest(request: WebLLMRequest, toolChoice?: 'none') {
  if (
    request.constrainedToolCalls !== undefined &&
    typeof request.constrainedToolCalls !== 'boolean'
  )
    throw new Error('constrainedToolCalls must be a boolean.')
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
  const native = { ...request }
  delete native.constrainedToolCalls
  const body = native.extra_body
  if (!body) return native as Omit<T, 'constrainedToolCalls'>
  const omitThinking = body.enable_thinking === false && /^gemma-4-/i.test(modelId)
  if (!('preserve_thinking' in body) && !omitThinking)
    return native as Omit<T, 'constrainedToolCalls'>
  const extra_body = { ...body }
  delete extra_body.preserve_thinking
  if (omitThinking) delete extra_body.enable_thinking
  return { ...native, extra_body } as Omit<T, 'constrainedToolCalls'>
}
