export const defaultEmbeddingBatchSize = 32

export function validateEmbeddingTexts(texts: readonly string[]): void {
  for (const text of texts) {
    if (typeof text !== 'string') throw new TypeError('WebLLM embedding inputs must be strings.')
    if (text === '') throw new Error('WebLLM cannot embed empty text.')
  }
}

export function validateEmbeddingBatchSize(batchSize: number): void {
  if (!Number.isInteger(batchSize) || batchSize < 1)
    throw new RangeError('batchSize must be a positive integer.')
}
