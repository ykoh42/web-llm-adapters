import type { ModelProfile } from '@langchain/core/language_models/profile'
import { isQwen, hasQwenToolLimit } from '../../../../shared/web-llm/qwen/models.js'
import { isQwen35 } from '../../../../shared/web-llm/qwen35/models.js'
import type { WebLLMRequest } from '../types.js'

const nativeToolModels = new Set([
  'Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC',
  'Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC',
  'Hermes-2-Pro-Mistral-7B-q4f16_1-MLC',
  'Hermes-3-Llama-3.1-8B-q4f32_1-MLC',
  'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
])

export function modelProfile(
  modelId: string,
  profile: ModelProfile = {},
  request?: WebLLMRequest,
): ModelProfile {
  for (const limit of [profile.maxInputTokens, profile.maxOutputTokens]) {
    if (limit !== undefined && (!Number.isSafeInteger(limit) || limit <= 0))
      throw new RangeError('Model token limits must be positive integers.')
  }
  if (profile.midConversationSystemMessages)
    throw new Error('WebLLM cannot send mid-conversation system messages.')
  if (profile.midConversationTools) throw new Error('WebLLM cannot add tools mid-conversation.')
  const gemma = /^gemma-4-/i.test(modelId)
  const qwen = isQwen(modelId) || isQwen35(modelId)
  const vision = modelId === 'Phi-3.5-vision-instruct-q4f16_1-MLC'
  const nativeTools = nativeToolModels.has(modelId)
  return {
    toolCalling: nativeTools || qwen || gemma,
    ...(!nativeTools && { structuredOutput: true }),
    ...((qwen || gemma) && { reasoningOutput: !modelId.startsWith('Qwen2.5-') }),
    ...((nativeTools || qwen || gemma) && {
      imageInputs: false,
      imageUrlInputs: false,
      // Deep Agents also uses this flag for tool audio, which these protocols cannot read.
      audioInputs: false,
    }),
    ...(vision && {
      imageInputs: true,
      imageUrlInputs: true,
      audioInputs: false,
      toolCalling: false,
      reasoningOutput: false,
      structuredOutput: true,
    }),
    ...structuredClone(profile),
    midConversationSystemMessages: false,
    midConversationTools: false,
    ...(gemma && request?.extra_body?.enable_thinking === true && { structuredOutput: false }),
    ...(hasQwenToolLimit(modelId) && {
      toolCalling: false,
    }),
    pdfInputs: false,
    videoInputs: false,
    fileMimeTypes: [],
    imageToolMessage: profile.imageToolMessage ?? false,
    pdfToolMessage: false,
    imageOutputs: false,
    audioOutputs: false,
    videoOutputs: false,
  }
}
