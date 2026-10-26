export const defaultEmbeddingBatchSize = 32

export function copyEmbeddingTexts(texts: readonly string[]): string[] {
  if (!Array.isArray(texts))
    throw new TypeError('WebLLM embedding inputs must be an array of strings.')
  const input = [...texts]
  for (const text of input) validateEmbeddingText(text)
  return input
}

export function validateEmbeddingText(text: string): void {
  if (typeof text !== 'string') throw new TypeError('WebLLM embedding inputs must be strings.')
  if (text === '') throw new Error('WebLLM cannot embed empty text.')
}

export function validateEmbeddingBatchSize(batchSize: number): void {
  if (!Number.isInteger(batchSize) || batchSize < 1)
    throw new RangeError('batchSize must be a positive integer.')
}
