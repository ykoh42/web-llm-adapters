import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { describe, expect, it, vi } from 'vitest'
import { WebLLMEmbeddingSearchStrategy, type EmbedTexts } from '../../src/index.js'

const encoder = new TextEncoder()

describe('WebLLMEmbeddingSearchStrategy', () => {
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

    expect(results.map(({ key }) => key)).toEqual(['cloud', 'tacos'])
    expect(results[0]?.score).toBe(1)
    expect(embed.mock.calls.map(([, type]) => type)).toEqual(['document', 'document', 'query'])
  })

  it('builds an index for content already in Strands storage', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage()
    await storage.write('existing', encoder.encode('stored before search'))

    const results = await strategy.search(storage, 'find it')

    expect(results.map(({ key }) => key)).toEqual(['existing'])
    expect(embed.mock.calls.map(([texts, type]) => [texts, type])).toEqual([
      [['find it'], 'query'],
      [['stored before search'], 'document'],
    ])
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
