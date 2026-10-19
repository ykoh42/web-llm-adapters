import type { ModelProfile } from '@langchain/core/language_models/profile'
import { isQwen } from '../../../../shared/web-llm/qwen/models.js'
import { isQwen35 } from '../../../../shared/web-llm/qwen35/models.js'

export function modelProfile(modelId: string, profile: ModelProfile = {}): ModelProfile {
  for (const limit of [profile.maxInputTokens, profile.maxOutputTokens]) {
    if (limit !== undefined && (!Number.isSafeInteger(limit) || limit <= 0))
      throw new RangeError('Model token limits must be positive integers.')
  }
  const gemma = /^gemma-4-/i.test(modelId)
  const qwen = isQwen(modelId) || isQwen35(modelId)
  const vision = modelId === 'Phi-3.5-vision-instruct-q4f16_1-MLC'
  return {
    ...(qwen || gemma
      ? {
          toolCalling: true,
          reasoningOutput: !modelId.startsWith('Qwen2.5-'),
          structuredOutput: true,
          imageInputs: false,
          imageUrlInputs: false,
          audioInputs: false,
        }
      : {}),
    ...(vision && {
      imageInputs: true,
      imageUrlInputs: true,
      audioInputs: false,
      toolCalling: false,
      reasoningOutput: false,
      structuredOutput: true,
    }),
    ...structuredClone(profile),
    ...(modelId === 'Qwen3-1.7B-q4f16_1-MLC' && { toolCalling: false }),
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
