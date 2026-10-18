import type { Storage } from '@strands-agents/sdk/storage'
import type { SearchStrategy, StorageSearchResult } from '@strands-agents/sdk/storage/search'

export type EmbeddingInputType = 'query' | 'document'

export type EmbedTexts = (
  texts: readonly string[],
  inputType: EmbeddingInputType,
) => Promise<readonly (readonly number[])[]>

export interface WebLLMEmbeddingSearchOptions {
  embed: EmbedTexts
  minScore?: number
}

interface IndexedText {
  content: string
  vector: readonly number[]
}

const decoder = new TextDecoder()

/** Use WebLLM vectors to rank a Strands storage or FileMemoryStore. */
export class WebLLMEmbeddingSearchStrategy implements SearchStrategy {
  readonly #embed: EmbedTexts
  readonly #minScore: number
  readonly #vectors = new Map<string, IndexedText>()
  #tail: Promise<void> = Promise.resolve()

  constructor(options: WebLLMEmbeddingSearchOptions) {
    if (
      options.minScore !== undefined &&
      (!Number.isFinite(options.minScore) || options.minScore < -1 || options.minScore > 1)
    )
      throw new RangeError('minScore must be between -1 and 1.')

    this.#embed = options.embed
    this.#minScore = options.minScore ?? -1
  }

  index(_storage: Storage, key: string, data: Uint8Array): Promise<void> {
    return this.#run(async () => {
      const content = decoder.decode(data).trim()
      if (!content) {
        this.#vectors.delete(key)
        return
      }

      const [vector] = this.#readVectors(await this.#embed([content], 'document'), 1)
      this.#vectors.set(key, { content, vector: vector! })
    })
  }

  search(storage: Storage, query: string): Promise<StorageSearchResult[]> {
    return this.#run(async () => {
      if (!query.trim()) return []

      const [queryVector] = this.#readVectors(await this.#embed([query.trim()], 'query'), 1)
      const keys = await storage.list('')
      const records = await Promise.all(
        keys.map(async (key) => {
          const data = await storage.read(key)
          if (!data) return undefined
          const content = decoder.decode(data).trim()
          if (!content) return undefined
          return { key, data, content }
        }),
      )
      const documents = records.filter((record) => record !== undefined)
      const missing = new Set<string>()

      for (const { key, content } of documents) {
        if (this.#vectors.get(key)?.content !== content) missing.add(content)
      }

      if (missing.size) {
        const texts = [...missing]
        const vectors = this.#readVectors(await this.#embed(texts, 'document'), texts.length)
        const byText = new Map(texts.map((text, index) => [text, vectors[index]!]))
        for (const { key, content } of documents) {
          if (this.#vectors.get(key)?.content !== content)
            this.#vectors.set(key, { content, vector: byText.get(content)! })
        }
      }

      const activeKeys = new Set(keys)
      for (const key of this.#vectors.keys()) {
        if (!activeKeys.has(key)) this.#vectors.delete(key)
      }

      return documents
        .map(({ key, data }) => ({
          key,
          data,
          score: cosine(queryVector!, this.#vectors.get(key)!.vector),
        }))
        .filter(({ score }) => score >= this.#minScore)
        .sort((left, right) => right.score - left.score)
    })
  }

  #readVectors(vectors: readonly (readonly number[])[], count: number) {
    if (vectors.length !== count)
      throw new TypeError(`Expected ${count} embeddings; received ${vectors.length}.`)

    for (const vector of vectors) {
      if (vector.length === 0 || !vector.every(Number.isFinite))
        throw new TypeError('Embeddings must be non-empty arrays of finite numbers.')
    }

    const dimensions = vectors[0]!.length
    if (vectors.some((vector) => vector.length !== dimensions))
      throw new TypeError('Embeddings in one batch must have the same dimensions.')

    return vectors
  }

  #run<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.#tail.then(operation, operation)
    this.#tail = result.then(
      () => undefined,
      () => undefined,
    )
    return result
  }
}

function cosine(left: readonly number[], right: readonly number[]) {
  if (left.length !== right.length)
    throw new TypeError('Query and document embeddings must have the same dimensions.')

  let dot = 0
  let leftSize = 0
  let rightSize = 0
  for (let index = 0; index < left.length; index++) {
    dot += left[index]! * right[index]!
    leftSize += left[index]! ** 2
    rightSize += right[index]! ** 2
  }

  if (leftSize === 0 || rightSize === 0) return 0
  return dot / Math.sqrt(leftSize * rightSize)
}
