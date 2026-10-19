import type { WebLLMModelConfig } from 'strands-web-llm'
import { sampling as nativeSampling } from '../../shared/sampling.js'

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
