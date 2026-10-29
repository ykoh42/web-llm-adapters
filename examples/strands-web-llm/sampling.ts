import type { WebLLMModelConfig } from 'strands-web-llm'
import type { ChatCompletionRequestBase } from '@mlc-ai/web-llm'

type Sampling = Pick<
  ChatCompletionRequestBase,
  'temperature' | 'top_p' | 'presence_penalty' | 'repetition_penalty' | 'extra_body'
>

/** Published model settings supported by WebLLM. top_k and min_p are not exposed. */
function nativeSampling(modelId: string, thinking: boolean): Sampling {
  const request = { repetition_penalty: 1, extra_body: { enable_thinking: thinking } }
  if (modelId.startsWith('Qwen3.5-'))
    return {
      ...request,
      temperature: thinking ? 1 : 0.7,
      top_p: thinking ? 0.95 : 0.8,
      presence_penalty: 1.5,
    }
  if (modelId.startsWith('Qwen3-'))
    return { ...request, temperature: thinking ? 0.6 : 0.7, top_p: thinking ? 0.95 : 0.8 }
  if (modelId.startsWith('Qwen2.5-'))
    return { temperature: 0.7, top_p: 0.8, repetition_penalty: 1.05 }
  if (modelId.startsWith('gemma-4-')) return { ...request, temperature: 1, top_p: 0.95 }
  if (modelId.startsWith('Phi-3.5-vision-')) return { temperature: 0 }
  if (modelId.startsWith('Hermes-3-Llama-3.1-'))
    return { temperature: 0.8, repetition_penalty: 1.1 }
  return thinking ? { extra_body: { enable_thinking: true } } : {}
}

export function sampling(
  modelId: string,
  thinking: boolean,
): Pick<WebLLMModelConfig, 'temperature' | 'topP' | 'request'> {
  const { temperature, top_p, ...request } = nativeSampling(modelId, thinking)
  return {
    ...(temperature != null && { temperature }),
    ...(top_p != null && { topP: top_p }),
    request,
  }
}
