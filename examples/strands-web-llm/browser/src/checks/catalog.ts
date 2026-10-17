import { prebuiltAppConfig, type ModelRecord } from '@mlc-ai/web-llm'

export const qwen = 'Qwen3-0.6B-q4f16_1-MLC'

export function record(id: string): ModelRecord {
  const model = prebuiltAppConfig.model_list.find((model) => model.model_id === id)
  if (!model) throw new Error(`Model not found: ${id}`)
  return structuredClone(model)
}
