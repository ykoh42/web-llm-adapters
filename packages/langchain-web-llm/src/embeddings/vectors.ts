import type { CreateEmbeddingResponse } from '@mlc-ai/web-llm'

export function readVectors(response: CreateEmbeddingResponse, count: number): number[][] {
  if (response.data.length !== count)
    throw new TypeError('WebLLM returned the wrong number of embeddings.')
  const vectors = new Array<number[]>(count)
  for (const item of response.data) {
    if (
      !Number.isInteger(item.index) ||
      item.index < 0 ||
      item.index >= count ||
      vectors[item.index] !== undefined
    )
      throw new TypeError('WebLLM returned invalid embedding indices.')
    if (!item.embedding.length || !item.embedding.every(Number.isFinite))
      throw new TypeError('WebLLM embeddings must contain finite numbers.')
    vectors[item.index] = item.embedding
  }
  return vectors
}

export function appendVectors(target: number[][], batch: number[][]): void {
  const dimensions = target[0]?.length ?? batch[0]?.length
  for (const vector of batch) {
    if (vector.length !== dimensions)
      throw new TypeError('WebLLM embeddings must have matching dimensions.')
    target.push(vector)
  }
}
