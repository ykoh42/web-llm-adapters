export const defaultEmbeddingBatchSize = 32

export function validateEmbeddingTexts(texts: readonly string[]): void {
  if (texts.some((text) => !text.trim())) throw new Error('WebLLM cannot embed empty text.')
}

export function validateEmbeddingBatchSize(batchSize: number): void {
  if (!Number.isInteger(batchSize) || batchSize < 1)
    throw new RangeError('batchSize must be a positive integer.')
}
