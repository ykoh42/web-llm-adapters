import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { describe, expect, it, vi } from 'vitest'
import { WebLLMEmbeddingSearchStrategy, type EmbedTexts } from '../../src/index.js'

const encoder = new TextEncoder()

describe('WebLLMEmbeddingSearchStrategy', () => {
  it('keeps non-unit embedding arrays unchanged across indexing and repeated searches', async () => {
    const query = Object.freeze([3, 4])
    const document = Object.freeze([6, 8])
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => texts.map(() => (type === 'query' ? query : document)),
      includeData: false,
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    await storage.write('note', encoder.encode('Stored note.'))

    for (let index = 0; index < 2; index++)
      expect(await storage.search('query')).toEqual([{ key: 'note', score: 1 }])
    expect(query).toEqual([3, 4])
    expect(document).toEqual([6, 8])
  })

  it('passes query and document text to the embedding model unchanged', async () => {
    const storage = new InMemoryStorage()
    const query = '  search query\n'
    const indexed = '\tindexed document\n'
    const stored = '  stored document  '
    await storage.write('stored', encoder.encode(stored))
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })

    await strategy.search(storage, query)
    await strategy.index(storage, 'indexed', encoder.encode(indexed))

    expect(embed).toHaveBeenCalledWith([query], 'query')
    expect(embed).toHaveBeenCalledWith([stored], 'document')
    expect(embed).toHaveBeenCalledWith([indexed], 'document')
  })

  it('uses cosine similarity for non-unit and zero vectors', async () => {
    const vectors = new Map([
      ['same', [6, 8]],
      ['partial', [0, 2]],
      ['other', [-4, 3]],
      ['zero', [0, 0]],
    ])
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) =>
        texts.map((text) => (type === 'query' && text !== 'zero' ? [3, 4] : vectors.get(text)!)),
      minScore: -1,
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    for (const key of vectors.keys()) await storage.write(key, encoder.encode(key))
    expect((await storage.search('query')).map(({ score }) => score)).toEqual([1, 0.8, 0, 0])
    expect((await storage.search('zero')).map(({ score }) => score)).toEqual([0, 0, 0, 0])
  })

  it('keeps cosine scores stable for very large and very small vectors', async () => {
    const vectors = new Map([
      ['large', [1e308, 1e308]],
      ['small', [1e-308, 1e-308]],
      ['other', [-1e308, 1e308]],
    ])
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) =>
        texts.map((text) =>
          type === 'query' && text === 'small query'
            ? [1e-308, 1e-308]
            : (vectors.get(text) ?? [1e308, 1e308]),
        ),
      minScore: -1,
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    for (const key of vectors.keys()) await storage.write(key, encoder.encode(key))

    for (const query of ['large query', 'small query']) {
      const results = await storage.search(query)
      expect(results.map(({ key }) => key)).toEqual(['large', 'small', 'other'])
      expect(results.map(({ score }) => score)).toEqual([1, 1, 0])
    }
  })

  it('includes results at the minimum cosine score', async () => {
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) =>
        texts.map((text) => (type === 'query' ? [1, 0] : text === 'boundary' ? [4, 3] : [0, 1])),
      minScore: 0.8,
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    await storage.write('boundary', encoder.encode('boundary'))
    await storage.write('below', encoder.encode('below'))

    await expect(storage.search('query')).resolves.toMatchObject([{ key: 'boundary', score: 0.8 }])
  })

  it('filters zero and negative matches by default', async () => {
    const vectors = new Map([
      ['match', [1, 0]],
      ['orthogonal', [0, 1]],
      ['opposite', [-1, 0]],
    ])
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) =>
        texts.map((text) =>
          type === 'query' ? (text === 'zero query' ? [0, 0] : [1, 0]) : vectors.get(text)!,
        ),
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    for (const key of vectors.keys()) await storage.write(key, encoder.encode(key))

    await expect(storage.search('query')).resolves.toMatchObject([{ key: 'match', score: 1 }])
    await expect(storage.search('zero query')).resolves.toEqual([])
  })

  it('rejects sparse embedding vectors', async () => {
    const sparse = new Array<number>(2)
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => texts.map(() => (type === 'query' ? [1, 0] : sparse)),
    })
    const storage = new InMemoryStorage()
    await storage.write('entry', encoder.encode('content'))

    await expect(strategy.search(storage, 'query')).rejects.toThrow('finite numbers')
  })

  it('indexes documents and ranks them by cosine similarity', async () => {
    const embed = vi.fn(async (texts: readonly string[], type: 'query' | 'document') =>
      texts.map((text) => {
        if (type === 'query' || text === 'cloud') return [1, 0]
        return [0, 1]
      }),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage({ searchStrategy: strategy })

    await storage.write('cloud', encoder.encode('cloud'))
    await storage.write('tacos', encoder.encode('tacos'))

    const results = await storage.search('what is snowflake?')

    expect(results.map(({ key }) => key)).toEqual(['cloud'])
    expect(results[0]?.data).toEqual(encoder.encode('cloud'))
    expect(results[0]?.score).toBe(1)
    expect(embed.mock.calls.map(([, type]) => type)).toEqual(['document', 'document', 'query'])
  })

  it('checks embedding output and dimension errors', async () => {
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => texts.map(() => (type === 'query' ? [1, 0] : [1])),
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    await storage.write('entry', encoder.encode('content'))

    await expect(storage.search('query')).rejects.toThrow('same dimensions')
    await expect(
      new WebLLMEmbeddingSearchStrategy({
        embed: async () => [[Number.NaN]],
      }).search(storage, 'query'),
    ).rejects.toThrow('finite numbers')
  })

  it.each([
    {
      name: 'a missing vector',
      embed: async (): Promise<number[][]> => [],
      message: 'Expected 1 embeddings; received 0.',
    },
    {
      name: 'an empty vector',
      embed: async (): Promise<number[][]> => [[]],
      message: 'non-empty arrays of finite numbers',
    },
  ])('rejects $name from the embedder', async ({ embed, message }) => {
    const storage = new InMemoryStorage()
    await storage.write('entry', encoder.encode('content'))

    await expect(
      new WebLLMEmbeddingSearchStrategy({ embed }).search(storage, 'query'),
    ).rejects.toThrow(message)
  })

  it('rejects inconsistent dimensions in a document batch', async () => {
    const embed: EmbedTexts = async (_texts, type) =>
      type === 'query'
        ? [[1, 0]]
        : [
            [1, 0],
            [0, 1, 0],
          ]
    const storage = new InMemoryStorage()
    await storage.write('first', encoder.encode('first'))
    await storage.write('second', encoder.encode('second'))

    await expect(
      new WebLLMEmbeddingSearchStrategy({ embed }).search(storage, 'query'),
    ).rejects.toThrow('same dimensions')
  })

  it('returns no results and skips embeddings for a blank query', async () => {
    const embed = vi.fn(async () => [[1]])
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })

    await expect(strategy.search(new InMemoryStorage(), '  ')).resolves.toEqual([])
    expect(embed).not.toHaveBeenCalled()
  })

  it('rejects an invalid minimum score', () => {
    expect(() => new WebLLMEmbeddingSearchStrategy({ embed: async () => [], minScore: 2 })).toThrow(
      'between -1 and 1',
    )
  })
})
