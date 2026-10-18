import type { WebLLMModelConfig } from 'strands-web-llm'

/** Model-card text settings supported by WebLLM. top_k and min_p are unavailable. */
export function sampling(
  modelId: string,
  thinking: boolean,
): Pick<WebLLMModelConfig, 'temperature' | 'topP' | 'request'> {
  const request = { repetition_penalty: 1, extra_body: { enable_thinking: thinking } }
  if (modelId.startsWith('Qwen3.5-'))
    return {
      temperature: 1,
      topP: thinking ? 0.95 : 1,
      request: { ...request, presence_penalty: thinking ? 1.5 : 2 },
    }
  if (modelId.startsWith('Qwen3-'))
    return { temperature: thinking ? 0.6 : 0.7, topP: thinking ? 0.95 : 0.8, request }
  if (modelId.startsWith('Qwen2.5-'))
    return { temperature: 0.7, topP: 0.8, request: { repetition_penalty: 1.05 } }
  if (modelId.startsWith('gemma-4-')) return { temperature: 1, topP: 0.95, request }
  return thinking ? { request: { extra_body: { enable_thinking: true } } } : {}
}
