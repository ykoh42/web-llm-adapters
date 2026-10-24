const modelIds = new Set([
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

export function isQwen(modelId: string) {
  return modelIds.has(modelId)
}
