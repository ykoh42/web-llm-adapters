import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { describe, expect, it, vi } from 'vitest'
import { WebLLMEmbeddingSearchStrategy, type EmbedTexts } from '../../src/index.js'

const encoder = new TextEncoder()

describe('WebLLMEmbeddingSearchStrategy', () => {
  it.each([false, true])(
    'keeps duplicate scores and tie order across different queries (data=%s)',
    async (includeData) => {
      const vectors = new Map([
        ['cloud', [1, 0]],
        ['bread', [0, 1]],
        ['zero', [0, 0]],
        ['opposite', [-1, 0]],
      ])
      const strategy = new WebLLMEmbeddingSearchStrategy({
        embed: async (texts) => texts.map((text) => vectors.get(text)!),
        includeData,
        minScore: -1,
      })
      const storage = new InMemoryStorage({ searchStrategy: strategy })
      const documents = new Map([
        ['a', 'cloud'],
        ['b', 'bread'],
        ['c', 'cloud'],
        ['d', 'cloud'],
      ])
      for (const [key, text] of documents) await storage.write(key, encoder.encode(text))

      for (const [query, order, scores] of [
        ['cloud', ['a', 'c', 'd', 'b'], [1, 1, 1, 0]],
        ['bread', ['b', 'a', 'c', 'd'], [1, 0, 0, 0]],
        ['zero', ['a', 'b', 'c', 'd'], [0, 0, 0, 0]],
        ['opposite', ['b', 'a', 'c', 'd'], [0, -1, -1, -1]],
      ] as const) {
        expect(await storage.search(query)).toEqual(
          order.map((key, index) => ({
            key,
            score: scores[index],
            ...(includeData && { data: encoder.encode(documents.get(key)!) }),
          })),
        )
      }
    },
  )

  it('reuses vectors for duplicate text until its last key is removed', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map((text) => (type === 'query' || text === 'shared' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, includeData: false, minScore: -1 })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    const shared = encoder.encode('shared')

    await storage.write('first', shared)
    await storage.write('second', shared)
    await storage.write('first', encoder.encode('changed'))
    await storage.write('third', shared)

    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(2)
    expect(await storage.search('query')).toEqual([
      { key: 'second', score: 1 },
      { key: 'third', score: 1 },
      { key: 'first', score: 0 },
    ])

    await storage.delete('second')
    await storage.delete('third')
    await storage.search('query')
    await storage.write('fourth', shared)

    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(3)
  })

  it.each([
    {
      mode: 'copy',
      expected: [
        { key: 'a', score: 1 },
        { key: 'c', score: 1 },
        { key: 'b', score: 0 },
      ],
    },
    {
      mode: 'rename',
      expected: [
        { key: 'c', score: 1 },
        { key: 'b', score: 0 },
      ],
    },
    {
      mode: 'replace',
      expected: [
        { key: 'a', score: 1 },
        { key: 'b', score: 1 },
      ],
    },
    {
      mode: 'swap',
      expected: [
        { key: 'b', score: 1 },
        { key: 'a', score: 0 },
      ],
    },
  ])('reuses cached text after a document $mode', async ({ mode, expected }) => {
    const storage = new InMemoryStorage()
    await storage.write('a', encoder.encode('cloud'))
    await storage.write('b', encoder.encode('bread'))
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map((text) => (type === 'query' || text === 'cloud' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, includeData: false, minScore: -1 })
    await strategy.search(storage, 'cloud')
    if (mode === 'rename') await storage.delete('a')
    if (mode === 'swap') {
      await storage.write('a', encoder.encode('bread'))
      await storage.write('b', encoder.encode('cloud'))
    } else await storage.write(mode === 'replace' ? 'b' : 'c', encoder.encode('cloud'))
    expect(await strategy.search(storage, 'cloud')).toEqual(expected)
    expect(
      embed.mock.calls.filter(([, type]) => type === 'document').map(([texts]) => texts),
    ).toEqual([['cloud', 'bread']])
  })

  it('refreshes copied text when the query dimension changes', async () => {
    const storage = new InMemoryStorage()
    await storage.write('a', encoder.encode('cloud'))
    let dimensions = 2
    const embed = vi.fn<EmbedTexts>(async (texts) =>
      texts.map(() => [1, ...Array<number>(dimensions - 1).fill(0)]),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, includeData: false })
    await strategy.search(storage, 'cloud')
    dimensions = 3
    await storage.write('b', encoder.encode('cloud'))
    expect(await strategy.search(storage, 'cloud')).toEqual([
      { key: 'a', score: 1 },
      { key: 'b', score: 1 },
    ])
    expect(
      embed.mock.calls.filter(([, type]) => type === 'document').map(([texts]) => texts),
    ).toEqual([['cloud'], ['cloud']])
  })

  it('keeps indexed document vectors when the embedder reuses its output array', async () => {
    const storage = new InMemoryStorage()
    const data = encoder.encode('Stored note.')
    await storage.write('note', data)
    const vector = [1, 0]
    const embed = vi.fn<EmbedTexts>(async (_texts, type) => {
      vector[0] = type === 'document' ? 1 : 0
      vector[1] = type === 'document' ? 0 : 1
      return [vector]
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    await strategy.index(storage, 'note', data)
    expect(await strategy.search(storage, 'query')).toEqual([])
    expect(await strategy.search(storage, 'query')).toEqual([])
    expect(embed.mock.calls.map(([, type]) => type)).toEqual(['document', 'query', 'query'])
  })

  it('keeps earlier batches when the embedder reuses document vectors', async () => {
    const storage = new InMemoryStorage()
    for (let index = 0; index < 64; index++)
      await storage.write(String(index).padStart(2, '0'), encoder.encode(`note ${index}`))
    const vectors = Array.from({ length: 32 }, () => [0, 0])
    const strategy = new WebLLMEmbeddingSearchStrategy({
      includeData: false,
      embed: async (texts, type) => {
        if (type === 'query') return [[1, 0]]
        texts.forEach((text, index) => {
          const first = Number(text.slice(5)) < 32
          vectors[index]![0] = first ? 1 : 0
          vectors[index]![1] = first ? 0 : 1
        })
        return vectors.slice(0, texts.length)
      },
    })
    const expected = Array.from({ length: 32 }, (_, index) => ({
      key: String(index).padStart(2, '0'),
      score: 1,
    }))
    expect(await strategy.search(storage, 'query')).toEqual(expected)
    expect(await strategy.search(storage, 'query')).toEqual(expected)
  })

  it('batches document embeddings and embeds duplicate content once', async () => {
    const storage = new InMemoryStorage()
    for (let index = 0; index < 70; index++)
      await storage.write(String(index).padStart(2, '0'), encoder.encode(`note ${index}`))
    await storage.write('duplicate', encoder.encode('note 0'))
    const batches: { type: string; texts: readonly string[] }[] = []
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => {
        batches.push({ type, texts })
        return texts.map(() => [1, 0])
      },
    })

    const results = await strategy.search(storage, 'query')

    expect(results).toHaveLength(71)
    expect(
      batches.filter(({ type }) => type === 'document').map(({ texts }) => texts.length),
    ).toEqual([32, 32, 6])
    expect(
      batches.filter(({ type }) => type === 'document').flatMap(({ texts }) => texts),
    ).toHaveLength(70)
  })

  it('skips inference for empty storage and drops vectors for removed documents', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage()
    await expect(strategy.search(storage, 'query')).resolves.toEqual([])
    expect(embed).not.toHaveBeenCalled()
    await storage.write('note', encoder.encode('hello'))
    await strategy.search(storage, 'query')
    expect(embed).toHaveBeenCalledTimes(2)
    await storage.write('note', encoder.encode(' '))
    await expect(strategy.search(storage, 'query')).resolves.toEqual([])
    expect(embed).toHaveBeenCalledTimes(2)
    await storage.write('note', encoder.encode('hello'))
    await strategy.search(storage, 'query')
    expect(embed).toHaveBeenCalledTimes(4)
  })

  it('drops cached vectors when a listed document becomes empty', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map(() => (type === 'query' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage()
    await storage.write('emptyable', encoder.encode('old content'))
    await storage.write('kept', encoder.encode('kept content'))
    await strategy.search(storage, 'query')

    await storage.write('emptyable', encoder.encode(' '))
    await strategy.search(storage, 'query')
    await storage.write('emptyable', encoder.encode('old content'))
    await strategy.search(storage, 'query')

    expect(
      embed.mock.calls.filter(([, type]) => type === 'document').map(([texts]) => texts.length),
    ).toEqual([2, 1])
  })

  it('refreshes document embeddings when text or whitespace changes', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    await storage.write('note', encoder.encode('hello'))
    await storage.write('note', encoder.encode(' hello '))
    expect(embed).toHaveBeenCalledTimes(2)
    await storage.write('note', encoder.encode('changed'))
    expect(embed).toHaveBeenCalledTimes(3)
    await storage.write('note', encoder.encode(' '))
    await storage.write('note', encoder.encode('changed'))
    expect(embed).toHaveBeenCalledTimes(4)
  })

  it('drops stale vectors when reindexing fails', async () => {
    const failure = new Error('Embedding failed')
    let failNextDocument = false
    const vectors = new Map([
      ['old content', [1, 0]],
      ['new content', [0, 1]],
    ])
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => {
        if (type === 'document' && failNextDocument) {
          failNextDocument = false
          throw failure
        }
        return texts.map((text) =>
          type === 'query' ? (text === 'new query' ? [0, 1] : [1, 0]) : vectors.get(text)!,
        )
      },
      includeData: false,
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    await storage.write('note', encoder.encode('old content'))
    await storage.search('old query')
    const read = vi.spyOn(storage, 'read')

    failNextDocument = true
    await expect(storage.write('note', encoder.encode('new content'))).rejects.toBe(failure)

    const results = await storage.search('new query')
    expect(read).toHaveBeenCalledOnce()
    expect(read).toHaveBeenCalledWith('note')
    expect(results).toMatchObject([{ key: 'note', score: 1 }])
  })

  it('retries a malformed document dimension without keeping a poisoned cache', async () => {
    const storage = new InMemoryStorage()
    await storage.write('note', encoder.encode('apple'))
    let failed = false
    const embed = vi.fn<EmbedTexts>(async (texts, type) => {
      if (type === 'document' && !failed) {
        failed = true
        return texts.map(() => [1, 0, 0])
      }
      return texts.map(() => [1, 0])
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    await expect(strategy.search(storage, 'fruit')).rejects.toThrow('same dimensions')
    expect(await strategy.search(storage, 'fruit')).toMatchObject([{ key: 'note', score: 1 }])
    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(2)
  })

  it('keeps valid batches when a later embedding batch has the wrong dimension', async () => {
    const storage = new InMemoryStorage()
    for (let index = 0; index < 33; index++)
      await storage.write(String(index), encoder.encode(`note ${index}`))
    let failed = false
    const batches: number[] = []
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => {
        if (type === 'document') {
          batches.push(texts.length)
          if (texts.length === 1 && !failed) {
            failed = true
            return [[1, 0, 0]]
          }
        }
        return texts.map(() => [1, 0])
      },
    })
    await expect(strategy.search(storage, 'notes')).rejects.toThrow('same dimensions')
    expect(await strategy.search(storage, 'notes')).toHaveLength(33)
    expect(batches).toEqual([32, 1, 1])
  })

  it('refreshes cached documents when the embedding dimension changes', async () => {
    const storage = new InMemoryStorage()
    await storage.write('note', encoder.encode('apple'))
    let dimensions = 2
    const embed = vi.fn<EmbedTexts>(async (texts) =>
      texts.map(() => Array.from({ length: dimensions }, (_, index) => (index === 0 ? 1 : 0))),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    expect(await strategy.search(storage, 'fruit')).toMatchObject([{ key: 'note', score: 1 }])
    dimensions = 3
    expect(await strategy.search(storage, 'fruit')).toMatchObject([{ key: 'note', score: 1 }])
    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(2)
  })

  it('refreshes an indexed document whose dimension differs from the query', async () => {
    const storage = new InMemoryStorage()
    const text = encoder.encode('apple')
    await storage.write('note', text)
    let indexed = false
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts) => {
        if (!indexed) {
          indexed = true
          return texts.map(() => [1, 0, 0])
        }
        return texts.map(() => [1, 0])
      },
    })
    await strategy.index(storage, 'note', text)
    expect(await strategy.search(storage, 'fruit')).toMatchObject([{ key: 'note', score: 1 }])
  })

  it('keeps a valid cache when one query has the wrong dimension', async () => {
    const storage = new InMemoryStorage()
    await storage.write('note', encoder.encode('apple'))
    let queries = 0
    let documents = 0
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => {
        if (type === 'query') {
          queries++
          return texts.map(() => (queries === 2 ? [1, 0, 0] : [1, 0]))
        }
        documents += texts.length
        return texts.map(() => [1, 0])
      },
    })
    expect(await strategy.search(storage, 'fruit')).toMatchObject([{ key: 'note', score: 1 }])
    await expect(strategy.search(storage, 'fruit')).rejects.toThrow('same dimensions')
    const before = documents
    expect(await strategy.search(storage, 'fruit')).toMatchObject([{ key: 'note', score: 1 }])
    expect(documents).toBe(before)
  })
})
