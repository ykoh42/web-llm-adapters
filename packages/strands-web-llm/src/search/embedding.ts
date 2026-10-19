import type { Storage } from '@strands-agents/sdk/storage'
import type { SearchStrategy, StorageSearchResult } from '@strands-agents/sdk/storage/search'

export type EmbeddingInputType = 'query' | 'document'

/** The callback may run concurrently for separate storage instances. */
export type EmbedTexts = (
  texts: readonly string[],
  inputType: EmbeddingInputType,
) => Promise<readonly (readonly number[])[]>

export interface WebLLMEmbeddingSearchOptions {
  embed: EmbedTexts
  /** Include scores at or above this value. Defaults to positive scores only. */
  minScore?: number
  /** Include document bytes in results. Defaults to true. */
  includeData?: boolean
}

interface IndexedText {
  content: string
  vector: readonly number[]
  scale: number
  sumSquares: number
}

const decoder = new TextDecoder()
const batchSize = 32

/** Use WebLLM vectors to rank a Strands storage or FileMemoryStore. */
export class WebLLMEmbeddingSearchStrategy implements SearchStrategy {
  readonly #embed: EmbedTexts
  readonly #minScore: number | undefined
  readonly #includeData: boolean
  readonly #stores = new WeakMap<Storage, Map<string, IndexedText>>()
  readonly #tails = new WeakMap<Storage, Promise<void>>()

  constructor(options: WebLLMEmbeddingSearchOptions) {
    if (
      options.minScore !== undefined &&
      (!Number.isFinite(options.minScore) || options.minScore < -1 || options.minScore > 1)
    )
      throw new RangeError('minScore must be between -1 and 1.')

    this.#embed = options.embed
    this.#minScore = options.minScore
    this.#includeData = options.includeData ?? true
  }

  index(storage: Storage, key: string, data: Uint8Array): Promise<void> {
    const content = decoder.decode(data).trim()
    return this.#run(storage, async () => {
      const vectors = this.#vectors(storage)
      if (!content) {
        vectors.delete(key)
        return
      }
      if (vectors.get(key)?.content === content) return

      vectors.delete(key)
      const [vector] = this.#readVectors(await this.#embed([content], 'document'), 1)
      vectors.set(key, { content, vector: vector!, ...vectorMetrics(vector!) })
    })
  }

  search(storage: Storage, query: string): Promise<StorageSearchResult[]> {
    return this.#run(storage, async () => {
      if (!query.trim()) return []
      const cached = this.#vectors(storage)

      const keys = await storage.list('')
      const documents: { key: string; data?: Uint8Array; content: string }[] = []
      for (let offset = 0; offset < keys.length; offset += batchSize) {
        // Bound open reads and finish the batch before reporting an error.
        const reads = keys
          .slice(offset, offset + batchSize)
          .map((key) => readText(storage, key, this.#includeData))
        const records = await Promise.all(reads).catch(async (error) => {
          await Promise.allSettled(reads)
          throw error
        })
        for (const record of records) if (record) documents.push(record)
      }
      if (!documents.length) {
        cached.clear()
        return []
      }

      const [queryVector] = this.#readVectors(await this.#embed([query.trim()], 'query'), 1)
      const queryMetrics = vectorMetrics(queryVector!)
      const normalizedQuery = queryVector!.map((value) =>
        queryMetrics.scale ? value / queryMetrics.scale : 0,
      )
      const missing = new Map<string, string[]>()

      for (const { key, content } of documents) {
        if (cached.get(key)?.content === content) continue
        const keys = missing.get(content)
        if (keys) keys.push(key)
        else missing.set(content, [key])
      }

      const entries = [...missing]
      for (let offset = 0; offset < entries.length; offset += batchSize) {
        const batch = entries.slice(offset, offset + batchSize)
        const vectors = this.#readVectors(
          await this.#embed(
            batch.map(([content]) => content),
            'document',
          ),
          batch.length,
        )
        batch.forEach(([content, keys], index) => {
          const vector = vectors[index]!
          const metrics = vectorMetrics(vector)
          for (const key of keys) cached.set(key, { content, vector, ...metrics })
        })
      }

      const activeKeys = new Set(documents.map(({ key }) => key))
      for (const key of cached.keys()) {
        if (!activeKeys.has(key)) cached.delete(key)
      }

      return documents
        .map(({ key, data }) => ({
          key,
          ...(data && { data }),
          score: cosine(normalizedQuery, queryMetrics.sumSquares, cached.get(key)!),
        }))
        .filter(({ score }) => (this.#minScore === undefined ? score > 0 : score >= this.#minScore))
        .sort((left, right) => right.score - left.score)
    })
  }

  #vectors(storage: Storage) {
    let vectors = this.#stores.get(storage)
    if (!vectors) {
      vectors = new Map<string, IndexedText>()
      this.#stores.set(storage, vectors)
    }
    return vectors
  }

  #readVectors(vectors: readonly (readonly number[])[], count: number) {
    if (vectors.length !== count)
      throw new TypeError(`Expected ${count} embeddings; received ${vectors.length}.`)

    for (const vector of vectors) {
      if (vector.length === 0)
        throw new TypeError('Embeddings must be non-empty arrays of finite numbers.')
      for (const value of vector) {
        if (!Number.isFinite(value))
          throw new TypeError('Embeddings must be non-empty arrays of finite numbers.')
      }
    }

    const dimensions = vectors[0]!.length
    if (vectors.some((vector) => vector.length !== dimensions))
      throw new TypeError('Embeddings in one batch must have the same dimensions.')

    return vectors
  }

  #run<T>(storage: Storage, operation: () => Promise<T>): Promise<T> {
    const tail = this.#tails.get(storage) ?? Promise.resolve()
    const result = tail.then(operation, operation)
    this.#tails.set(
      storage,
      result.then(
        () => undefined,
        () => undefined,
      ),
    )
    return result
  }
}

async function readText(storage: Storage, key: string, includeData: boolean) {
  const data = await storage.read(key)
  if (!data) return undefined
  const content = decoder.decode(data).trim()
  if (!content) return undefined
  return { key, content, ...(includeData && { data }) }
}

function vectorMetrics(vector: readonly number[]) {
  let scale = 0
  for (const value of vector) scale = Math.max(scale, Math.abs(value))

  let sumSquares = 0
  if (scale) {
    for (const value of vector) {
      const scaled = value / scale
      sumSquares += scaled * scaled
    }
  }
  return { scale, sumSquares }
}

function cosine(query: readonly number[], querySumSquares: number, document: IndexedText) {
  if (query.length !== document.vector.length)
    throw new TypeError('Query and document embeddings must have the same dimensions.')
  if (!querySumSquares || !document.scale) return 0

  let dot = 0
  for (let index = 0; index < query.length; index++)
    dot += query[index]! * (document.vector[index]! / document.scale)

  const score = dot / Math.sqrt(querySumSquares * document.sumSquares)
  return Math.max(-1, Math.min(1, score))
}
