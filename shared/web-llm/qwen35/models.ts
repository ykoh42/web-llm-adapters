const modelIds = new Set([
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

export function isQwen35(modelId: string) {
  return modelIds.has(modelId)
}
