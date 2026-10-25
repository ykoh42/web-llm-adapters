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
  sumSquares: number
  references: number
}

interface StorageIndex {
  byKey: Map<string, IndexedText>
  byContent: Map<string, IndexedText>
}

type QueryEmbeddingResult = { vector: number[] } | { error: unknown }

const decoder = new TextDecoder('utf-8', { fatal: true })
const batchSize = 32

/** Use WebLLM vectors to rank UTF-8 text in a Strands storage or FileMemoryStore. */
export class WebLLMEmbeddingSearchStrategy implements SearchStrategy {
  readonly #embed: EmbedTexts
  readonly #minScore: number | undefined
  readonly #includeData: boolean
  readonly #stores = new WeakMap<Storage, StorageIndex>()
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
    const content = decodeText(data)
    return this.#run(storage, async () => {
      const index = this.#cache(storage)
      if (!content?.trim()) {
        this.#remove(index, key)
        return
      }
      if (index.byKey.get(key)?.content === content) return

      this.#remove(index, key)
      const cached = index.byContent.get(content)
      if (cached) {
        this.#assign(index, key, cached)
        return
      }

      const [vector] = this.#readVectors(await this.#embed([content], 'document'), 1)
      this.#assign(index, key, {
        content,
        vector: vector!,
        sumSquares: scaleVector(vector!),
        references: 0,
      })
    })
  }

  search(storage: Storage, query: string): Promise<StorageSearchResult[]> {
    return this.#run(storage, async () => {
      if (!query.trim()) return []
      const index = this.#cache(storage)
      const cached = index.byKey

      const keys = await storage.list('')
      const documents: { key: string; data?: Uint8Array; content: string }[] = []
      let queryEmbedding: Promise<QueryEmbeddingResult> | undefined
      const startQueryEmbedding = () => {
        queryEmbedding ??= Promise.resolve()
          .then(() => this.#embed([query], 'query'))
          .then((vectors) => this.#readVectors(vectors, 1)[0]!)
          .then(
            (vector) => ({ vector }),
            (error: unknown) => ({ error }),
          )
        return queryEmbedding
      }

      for (let offset = 0; offset < keys.length; offset += batchSize) {
        // Bound open reads and finish the batch before reporting an error.
        const reads = keys.slice(offset, offset + batchSize).map((key) =>
          readText(storage, key, this.#includeData).then((record) => {
            if (record) startQueryEmbedding()
            return record
          }),
        )
        const records = await Promise.all(reads).catch(async (error) => {
          await Promise.allSettled(reads)
          if (queryEmbedding) await queryEmbedding
          throw error
        })
        for (const record of records) if (record) documents.push(record)
      }
      if (!documents.length) {
        index.byKey.clear()
        index.byContent.clear()
        return []
      }

      const queryResult = await startQueryEmbedding()
      if ('error' in queryResult) throw queryResult.error
      const queryVector = queryResult.vector
      const querySumSquares = scaleVector(queryVector)
      const missing = new Map<string, string[]>()

      for (const { key, content } of documents) {
        const previous = cached.get(key)
        if (previous?.content === content && previous.vector.length === queryVector.length) continue
        const keys = missing.get(content)
        if (keys) keys.push(key)
        else missing.set(content, [key])
      }

      const reusable = [...missing].flatMap(([content, keys]) => {
        const previous = index.byContent.get(content)
        return previous?.vector.length === queryVector.length ? [{ content, keys, previous }] : []
      })
      for (const { content, keys, previous } of reusable) {
        for (const key of keys) this.#assign(index, key, previous)
        missing.delete(content)
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
          queryVector.length,
        )
        batch.forEach(([content, keys], offset) => {
          const vector = vectors[offset]!
          const entry = {
            content,
            vector,
            sumSquares: scaleVector(vector),
            references: 0,
          }
          for (const key of keys) this.#assign(index, key, entry)
        })
      }

      const activeKeys = new Set(documents.map(({ key }) => key))
      for (const key of cached.keys()) {
        if (!activeKeys.has(key)) this.#remove(index, key)
      }

      const scores =
        index.byContent.size < documents.length ? new Map<IndexedText, number>() : undefined
      return documents
        .map(({ key, data }) => {
          const document = cached.get(key)!
          const score = scores?.get(document) ?? cosine(queryVector, querySumSquares, document)
          if (document.references > 1) scores?.set(document, score)
          return { key, ...(data && { data }), score }
        })
        .filter(({ score }) => (this.#minScore === undefined ? score > 0 : score >= this.#minScore))
        .sort((left, right) => right.score - left.score)
    })
  }

  #cache(storage: Storage) {
    let index = this.#stores.get(storage)
    if (!index) {
      index = { byKey: new Map(), byContent: new Map() }
      this.#stores.set(storage, index)
    }
    return index
  }

  #assign(index: StorageIndex, key: string, entry: IndexedText) {
    if (index.byKey.get(key) === entry) return
    this.#remove(index, key)
    entry.references++
    index.byKey.set(key, entry)
    index.byContent.set(entry.content, entry)
  }

  #remove(index: StorageIndex, key: string) {
    const entry = index.byKey.get(key)
    if (!entry) return
    index.byKey.delete(key)
    entry.references--
    if (entry.references === 0 && index.byContent.get(entry.content) === entry)
      index.byContent.delete(entry.content)
  }

  #readVectors(vectors: readonly (readonly number[])[], count: number, dimensions?: number) {
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

    const size = vectors[0]!.length
    if (vectors.some((vector) => vector.length !== size))
      throw new TypeError('Embeddings in one batch must have the same dimensions.')
    if (dimensions !== undefined && size !== dimensions)
      throw new TypeError('Query and document embeddings must have the same dimensions.')

    // The embedder may reuse its arrays on the next request.
    return vectors.map((vector) => [...vector])
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
  const content = decodeText(data)
  if (!content?.trim()) return undefined
  return { key, content, ...(includeData && { data }) }
}

function decodeText(data: Uint8Array) {
  try {
    return decoder.decode(data)
  } catch {
    return undefined
  }
}

// Scale private copies once instead of on every cached search.
function scaleVector(vector: number[]) {
  let scale = 0
  for (const value of vector) scale = Math.max(scale, Math.abs(value))

  let sumSquares = 0
  for (let index = 0; index < vector.length; index++) {
    const value = scale ? vector[index]! / scale : 0
    vector[index] = value
    sumSquares += value * value
  }
  return sumSquares
}

function cosine(query: readonly number[], querySumSquares: number, document: IndexedText) {
  if (query.length !== document.vector.length)
    throw new TypeError('Query and document embeddings must have the same dimensions.')
  if (!querySumSquares || !document.sumSquares) return 0

  let dot = 0
  for (let index = 0; index < query.length; index++) dot += query[index]! * document.vector[index]!

  const score = dot / Math.sqrt(querySumSquares * document.sumSquares)
  return Math.max(-1, Math.min(1, score))
}
