import type { Api, Model } from '@earendil-works/pi-ai'
import type { WebLLMRequest, WebLLMStreamOptions } from './types.js'

export function requestOptions(
  model: Model<Api>,
  defaults: WebLLMRequest,
  call: WebLLMStreamOptions,
  simple: boolean,
): WebLLMRequest {
  if (call.deferred) throw new Error('WebLLM does not support deferred requests.')
  if (call.thinkingBudgets) throw new Error('WebLLM does not support thinking token budgets.')
  const request = structuredClone({
    ...defaults,
    ...call.request,
    ...(call.temperature !== undefined && { temperature: call.temperature }),
    max_tokens:
      call.maxTokens ?? call.request?.max_tokens ?? defaults.max_tokens ?? model.maxTokens,
    ...model.samplingParams,
    ...call.samplingParams,
  })
  for (const key of [
    'model',
    'messages',
    'prompt',
    'tools',
    'tool_choice',
    'n',
    'stream',
    'stream_options',
  ])
    if (key in request) throw new Error(`Use the provider API instead of overriding ${key}.`)
  positiveInteger(request.max_tokens, 'maxTokens')
  if (call.reasoning !== undefined && !model.reasoning)
    throw new Error('This WebLLM model does not support reasoning.')
  if (call.reasoning !== undefined || (simple && model.reasoning))
    request.extra_body = {
      ...request.extra_body,
      enable_thinking: call.reasoning !== undefined,
    }
  return request
}

export function positiveInteger(value: number, name: string) {
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error(`${name} must be a positive integer.`)
}
