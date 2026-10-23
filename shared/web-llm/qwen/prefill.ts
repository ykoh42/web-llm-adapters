import { isQwen } from './models.js'
import { isQwen35 } from '../qwen35/models.js'

export function checkThinkingPrefill(modelId: string, prefilled?: boolean) {
  if (prefilled !== undefined && typeof prefilled !== 'boolean')
    throw new TypeError('thinkingPrefilled must be a boolean.')
  if (prefilled && !(isQwen35(modelId) || (modelId.startsWith('Qwen3-') && isQwen(modelId))))
    throw new Error('thinkingPrefilled requires a Qwen thinking model.')
}
