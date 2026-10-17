import type { WebLLMModelConfig } from './types.js'

/** SDK settings take precedence over their native WebLLM equivalents. */
export function generationOptions(config: WebLLMModelConfig) {
  const options = {
    ...config.request,
    model: config.modelId,
    stream: true as const,
    stream_options: { include_usage: true },
    n: 1,
  }
  if (config.maxTokens !== undefined) options.max_tokens = config.maxTokens
  if (config.temperature !== undefined) options.temperature = config.temperature
  if (config.topP !== undefined) options.top_p = config.topP
  return options
}
