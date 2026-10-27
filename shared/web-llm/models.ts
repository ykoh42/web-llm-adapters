const qwenIds = new Set([
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f32_1-MLC',
  'Qwen2.5-0.5B-Instruct-q0f16-MLC',
  'Qwen2.5-0.5B-Instruct-q0f32-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f32_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f32_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f32_1-MLC',
  'Qwen3-0.6B-q4f16_1-MLC',
  'Qwen3-0.6B-q4f32_1-MLC',
  'Qwen3-0.6B-q0f16-MLC',
  'Qwen3-0.6B-q0f32-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3-1.7B-q4f32_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f32_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen3-8B-q4f32_1-MLC',
])

const qwen35Ids = new Set([
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'Qwen3.5-0.8B-q4f32_1-MLC',
  'Qwen3.5-0.8B-q0f16-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f32_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f32_1-MLC',
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-9B-q4f32_1-MLC',
])

const constrainedToolIds = new Set([
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
])

const visionIds = new Set([
  'Phi-3.5-vision-instruct-q4f16_1-MLC',
  'Phi-3.5-vision-instruct-q4f32_1-MLC',
])

export function isQwen(modelId: string) {
  return qwenIds.has(modelId)
}

export function isQwen35(modelId: string) {
  return qwen35Ids.has(modelId)
}

export function isGemma(modelId: string) {
  return /^gemma-4-/i.test(modelId)
}

export function supportsConstrainedToolCalls(modelId: string) {
  return constrainedToolIds.has(modelId)
}

export function isVisionModel(modelId: string) {
  return visionIds.has(modelId)
}
