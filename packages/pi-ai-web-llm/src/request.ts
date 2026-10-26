import type { Api, Model, TranscriptContext } from '@earendil-works/pi-ai'
import {
  clampMaxTokensToContext,
  resolveSamplingParams,
} from '@earendil-works/pi-ai/api/simple-options'
import { clampThinkingLevel } from '@earendil-works/pi-ai/models'
import { normalizeResponseFormat } from '../../../shared/web-llm/response.js'
import { checkRequest, preserveThinkingRequest } from '../../../shared/web-llm/request.js'
import type { WebLLMRequest, WebLLMStreamOptions } from './types.js'

export function requestOptions(
  model: Model<Api>,
  defaults: WebLLMRequest,
  call: WebLLMStreamOptions,
  simple: boolean,
  context: TranscriptContext,
): WebLLMRequest {
  if (
    call.toolChoice !== undefined &&
    call.toolChoice !== 'auto' &&
    call.toolChoice !== 'none' &&
    (!call.toolChoice ||
      call.toolChoice.type !== 'function' ||
      typeof call.toolChoice.function?.name !== 'string' ||
      !call.toolChoice.function.name)
  )
    throw new Error('toolChoice must be auto, none, or a function choice.')
  if (call.deferred) throw new Error('WebLLM does not support deferred requests.')
  if (Object.values(call.thinkingBudgets ?? {}).some((budget) => budget !== undefined))
    throw new Error('WebLLM does not support thinking token budgets.')
  const configuredMaxTokens =
    call.request?.max_tokens !== undefined ? call.request.max_tokens : defaults.max_tokens
  let maxTokens =
    call.maxTokens ?? (configuredMaxTokens === undefined ? model.maxTokens : configuredMaxTokens)
  if (simple && typeof maxTokens === 'number' && Number.isSafeInteger(maxTokens) && maxTokens > 0)
    maxTokens = clampMaxTokensToContext(model, context, maxTokens)
  const sampling = resolveSamplingParams(model, call.reasoning ?? 'off', call.samplingParams)
  const request = structuredClone({
    ...defaults,
    ...call.request,
    ...(call.temperature !== undefined && { temperature: call.temperature }),
    max_tokens: maxTokens,
    ...sampling,
  })
  preserveThinkingRequest(
    request,
    sampling?.extra_body,
    call.request?.extra_body,
    defaults.extra_body,
  )
  checkRequest(request)
  if (request.max_tokens != null) positiveInteger(request.max_tokens, 'maxTokens')
  if (call.reasoning !== undefined && !model.reasoning)
    throw new Error('This WebLLM model does not support reasoning.')
  // Explicit sampling bodies take precedence over named thinking controls.
  if (
    (call.reasoning !== undefined || (simple && model.reasoning)) &&
    (!sampling || !Object.hasOwn(sampling, 'extra_body'))
  )
    request.extra_body = {
      ...request.extra_body,
      enable_thinking: clampThinkingLevel(model, call.reasoning ?? 'off') !== 'off',
    }
  if (request.response_format)
    request.response_format = normalizeResponseFormat(request.response_format)
  return request
}

export function positiveInteger(value: number, name: string) {
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error(`${name} must be a positive integer.`)
}
