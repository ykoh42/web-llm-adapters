import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { describe, expect, it, vi } from 'vitest'
import { WebLLMEmbeddingSearchStrategy, type EmbedTexts } from '../../src/index.js'
import { pause } from '../../fixtures/engine.js'

const encoder = new TextEncoder()

describe('WebLLMEmbeddingSearchStrategy', () => {
  it('finishes open reads before reporting a failure and can search again', async () => {
    const storage = new InMemoryStorage()
    await storage.write('bad', encoder.encode('bad note'))
    await storage.write('slow', encoder.encode('slow note'))
    const hold = pause()
    const failure = new Error('Read failed')
    const read = storage.read.bind(storage)
    const reading = vi.spyOn(storage, 'read').mockImplementation(async (key) => {
      if (key === 'bad') throw failure
      await hold.wait
      return read(key)
    })
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    let settled = false
    const failed = expect(strategy.search(storage, 'notes')).rejects.toBe(failure)
    void failed.then(() => {
      settled = true
    })
    await vi.waitFor(() => expect(reading).toHaveBeenCalledTimes(2))
    expect(settled).toBe(false)
    expect(embed).not.toHaveBeenCalled()
    hold.resume()
    await failed
    reading.mockRestore()
    expect(await strategy.search(storage, 'notes')).toHaveLength(2)
  })

  it('searches large stores without opening every read at once', async () => {
    const storage = new InMemoryStorage()
    for (let index = 0; index < 256; index++)
      await storage.write(String(index).padStart(3, '0'), encoder.encode(`note ${index}`))
    const read = storage.read.bind(storage)
    let active = 0
    let peak = 0
    vi.spyOn(storage, 'read').mockImplementation(async (key) => {
      active++
      peak = Math.max(peak, active)
      try {
        if (active > 32) throw new Error('Too many open reads')
        await new Promise((resolve) => setTimeout(resolve, 0))
        return await read(key)
      } finally {
        active--
      }
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts) => texts.map(() => [1, 0]),
    })
    const results = await strategy.search(storage, 'notes')
    expect(results.map(({ key }) => key)).toEqual(await storage.list(''))
    expect(peak).toBeLessThanOrEqual(32)
    expect(active).toBe(0)
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
      storage: new InMemoryStorage().namespace(''),
      search: strategy,
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

  it('does not queue searches for different storages', async () => {
    const hold = pause()
    const slowStarted = pause()
    const embed = vi.fn<EmbedTexts>(async (texts, type) => {
      if (type === 'query' && texts[0] === 'slow query') {
        slowStarted.resume()
        await hold.wait
      }
      return texts.map(() => [1, 0])
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const first = new InMemoryStorage()
    const second = new InMemoryStorage()
    await first.write('note', encoder.encode('first'))
    await second.write('note', encoder.encode('second'))

    const slow = strategy.search(first, 'slow query')
    let fast: ReturnType<typeof strategy.search> | undefined
    let fastFinished = false
    try {
      await slowStarted.wait
      fast = strategy.search(second, 'fast query')
      void fast.then(
        () => {
          fastFinished = true
        },
        () => undefined,
      )
      await vi.waitFor(() => expect(fastFinished).toBe(true))
    } finally {
      hold.resume()
    }

    await expect(slow).resolves.toHaveLength(1)
    await expect(fast).resolves.toHaveLength(1)
  })

  it('shares document embeddings across concurrent searches of one storage', async () => {
    const hold = pause()
    const documentStarted = pause()
    const embed = vi.fn<EmbedTexts>(async (texts, type) => {
      if (type === 'document') {
        documentStarted.resume()
        await hold.wait
      }
      return texts.map(() => [1, 0])
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage()
    await storage.write('note', encoder.encode('shared note'))

    const first = strategy.search(storage, 'first query')
    await documentStarted.wait
    const second = strategy.search(storage, 'second query')
    hold.resume()
    await expect(Promise.all([first, second])).resolves.toHaveLength(2)
    expect(embed.mock.calls.filter(([, type]) => type === 'document')).toHaveLength(1)
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

  it('reuses unchanged document embeddings and refreshes changed content', async () => {
    const embed = vi.fn<EmbedTexts>(async (texts) => texts.map(() => [1, 0]))
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    await storage.write('note', encoder.encode('hello'))
    await storage.write('note', encoder.encode(' hello '))
    expect(embed).toHaveBeenCalledTimes(1)
    await storage.write('note', encoder.encode('changed'))
    expect(embed).toHaveBeenCalledTimes(2)
    await storage.write('note', encoder.encode(' '))
    await storage.write('note', encoder.encode('changed'))
    expect(embed).toHaveBeenCalledTimes(3)
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

  it('snapshots queued bytes before waiting for earlier indexes', async () => {
    const hold = pause()
    const firstEmbeddingStarted = pause()
    const embed = vi.fn<EmbedTexts>(async (texts, type) => {
      if (type === 'document' && texts[0] === 'first note') {
        firstEmbeddingStarted.resume()
        await hold.wait
      }
      return texts.map(() => [1, 0])
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({ embed })
    const storage = new InMemoryStorage({ searchStrategy: strategy })
    const firstWrite = storage.write('first', encoder.encode('first note'))
    await firstEmbeddingStarted.wait

    const input = encoder.encode('second note')
    const secondWrite = storage.write('second', input)
    input.set(encoder.encode('other note'))
    hold.resume()
    await Promise.all([firstWrite, secondWrite])

    expect(embed.mock.calls).toEqual([
      [['first note'], 'document'],
      [['second note'], 'document'],
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
