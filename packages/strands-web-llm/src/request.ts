import type { WebLLMModelConfig } from './types.js'
import { isTextFormat, normalizeResponseFormat } from '../../../shared/web-llm/response.js'
import { nativeRequest } from '../../../shared/web-llm/request.js'

export function hasStructuredResponse(config: WebLLMModelConfig) {
  return !isTextFormat(config.request?.response_format)
}

/** SDK settings take precedence over their native WebLLM equivalents. */
export function generationOptions(config: WebLLMModelConfig) {
  const options = {
    ...nativeRequest(config.request ?? {}, config.modelId),
    model: config.modelId,
    stream: true as const,
    stream_options: { include_usage: true },
    n: 1,
  }
  if (options.response_format)
    options.response_format = normalizeResponseFormat(options.response_format)
  if (config.maxTokens !== undefined) options.max_tokens = config.maxTokens
  if (config.temperature !== undefined) options.temperature = config.temperature
  if (config.topP !== undefined) options.top_p = config.topP
  return options
}
