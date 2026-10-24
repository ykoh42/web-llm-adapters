import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { describe, expect, it, vi } from 'vitest'
import { WebLLMEmbeddingSearchStrategy, type EmbedTexts } from '../../src/index.js'

const encoder = new TextEncoder()

describe('WebLLMEmbeddingSearchStrategy', () => {
  it('skips invalid UTF-8 entries without failing storage writes', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, includeData: false })
    const storage = new InMemoryStorage({ searchStrategy: strategy })

    await storage.write('binary', new Uint8Array([0xff, 0xfe]))
    await storage.write('text', encoder.encode('plain text'))

    expect(await storage.read('binary')).toEqual(new Uint8Array([0xff, 0xfe]))
    expect(await storage.search('query')).toEqual([{ key: 'text', score: 1 }])
    expect(embed.mock.calls).toEqual([
      [['plain text'], 'document'],
      [['query'], 'query'],
    ])

    const invalidText = new Uint8Array([0xff])
    await storage.write('text', invalidText)
    expect(await storage.read('text')).toEqual(invalidText)
    expect(await storage.search('query')).toEqual([])
  })

  it('can omit document bytes from search results', async () => {
    const storage = new InMemoryStorage()
    await storage.write('note', encoder.encode('Short note.'))
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts) => texts.map(() => [1, 0]),
      includeData: false,
    })

    await expect(strategy.search(storage, 'query')).resolves.toEqual([{ key: 'note', score: 1 }])
  })

  it('checks current text when results omit data', async () => {
    const storage = new InMemoryStorage()
    const oldText = encoder.encode('Old note.')
    const newText = encoder.encode('New note.')
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) =>
        texts.map((text) => (type === 'query' || text === 'New note.' ? [0, 1] : [1, 0])),
      includeData: false,
    })
    await strategy.index(storage, 'note', oldText)
    const read = vi.spyOn(storage, 'read')

    await storage.write('note', newText)
    const results = await strategy.search(storage, 'new query')

    expect(read).toHaveBeenCalledOnce()
    expect(read).toHaveBeenCalledWith('note')
    expect(results).toMatchObject([{ key: 'note', score: 1 }])
  })

  it('reuses document vectors through Strands FileMemoryStore', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map(() => (type === 'query' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, includeData: false })
    const memory = new FileMemoryStore({
      name: 'notes',
      storage: new InMemoryStorage({ searchStrategy: strategy }).namespace(''),
    })
    await memory.add('# Cloud\nData Cloud')

    await memory.search('cloud')
    await memory.search('cloud')

    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(1)
    expect(embed.mock.calls.filter(([, type]) => type === 'query')).toHaveLength(2)
  })

  it('keeps cached vectors for each storage when a strategy is shared', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map((text) => (type === 'query' || text === 'cloud' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, minScore: -1 })
    const first = new InMemoryStorage({ searchStrategy: strategy })
    const second = new InMemoryStorage({ searchStrategy: strategy })
    await first.write('note', encoder.encode('cloud'))
    await second.write('note', encoder.encode('bread'))
    for (let round = 0; round < 3; round++) {
      expect((await first.search('cloud'))[0]?.score).toBe(1)
      expect((await second.search('cloud'))[0]?.score).toBe(0)
    }
    await first.delete('note')
    expect(await first.search('cloud')).toEqual([])
    expect((await second.search('cloud'))[0]?.score).toBe(0)
    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(2)
  })

  it('keeps cached vectors separate for namespaces that reuse a key', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map((text) => (type === 'query' || text === 'cloud' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, minScore: -1 })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    const cloud = storage.namespace('cloud')
    const bread = storage.namespace('bread')
    await cloud.write('note', encoder.encode('cloud'))
    await bread.write('note', encoder.encode('bread'))

    for (let round = 0; round < 3; round++) {
      expect((await cloud.search('query'))[0]?.score).toBe(1)
      expect((await bread.search('query'))[0]?.score).toBe(0)
    }
    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toEqual([
      [['cloud'], 'document'],
      [['bread'], 'document'],
    ])
  })

  it('refreshes a namespace after its parent overwrites the same entry', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map((text) => (type === 'query' || text === 'new' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed, includeData: false, minScore: -1 })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    const namespace = storage.namespace('memory')
    await namespace.write('note', encoder.encode('old'))
    expect(await namespace.search('query')).toEqual([{ key: 'note', score: 0 }])

    await storage.write('memory/note', encoder.encode('new'))

    expect(await namespace.search('query')).toEqual([{ key: 'note', score: 1 }])
  })

  it('keeps equal relative keys separate across namespaces', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts, type) =>
      texts.map((text) => (type === 'query' || text === 'left' ? [1, 0] : [0, 1])),
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed,
      includeData: false,
      minScore: -1,
    })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    const left = storage.namespace('left')
    const right = storage.namespace('right')
    await left.write('note', encoder.encode('left'))
    await right.write('note', encoder.encode('right'))

    expect(await left.search('query')).toEqual([{ key: 'note', score: 1 }])
    expect(await right.search('query')).toEqual([{ key: 'note', score: 0 }])
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
})
