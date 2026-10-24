import { InMemoryStorage } from '@strands-agents/sdk/storage'
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

  it('embeds the query while reading the remaining documents', async () => {
    const storage = new InMemoryStorage()
    await storage.write('first', encoder.encode('first'))
    await storage.write('second', encoder.encode('second'))
    const readHold = pause()
    const queryHold = pause()
    const read = storage.read.bind(storage)
    let secondReadStarted = false
    let secondReadDone = false
    vi.spyOn(storage, 'read').mockImplementation(async (key) => {
      if (key === 'second') {
        secondReadStarted = true
        await readHold.wait
        secondReadDone = true
      }
      return read(key)
    })
    let queryStartedEmbedding = false
    const embed = vi.fn<EmbedTexts>(async (texts, type) => {
      if (type === 'query') {
        queryStartedEmbedding = true
        await queryHold.wait
      }
      return texts.map((text) => (text === 'first' ? [1, 0] : [0, 1]))
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed,
      includeData: false,
      minScore: -1,
    })
    let settled = false
    const pending = strategy.search(storage, 'first').then(
      (results) => {
        settled = true
        return { results }
      },
      (error: unknown) => {
        settled = true
        return { error }
      },
    )

    try {
      await vi.waitFor(() => expect(queryStartedEmbedding).toBe(true))
      expect(secondReadStarted).toBe(true)
      expect(secondReadDone).toBe(false)
      expect(settled).toBe(false)
    } finally {
      readHold.resume()
      queryHold.resume()
    }

    const outcome = await pending
    if ('error' in outcome) throw outcome.error
    expect(outcome.results).toEqual([
      { key: 'first', score: 1 },
      { key: 'second', score: 0 },
    ])
  })

  it('searches large stores without opening every read at once', async () => {
    const storage = new InMemoryStorage()
    for (let index = 0; index < 256; index++)
      await storage.write(String(index).padStart(3, '0'), encoder.encode(`note ${index}`))
    const read = storage.read.bind(storage)
    const hold = pause()
    let active = 0
    let peak = 0
    vi.spyOn(storage, 'read').mockImplementation(async (key) => {
      active++
      peak = Math.max(peak, active)
      try {
        if (active > 32) throw new Error('Too many open reads')
        await hold.wait
        return await read(key)
      } finally {
        active--
      }
    })
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts) => texts.map(() => [1, 0]),
    })
    const pending = strategy.search(storage, 'notes').then(
      (results) => ({ results }),
      (error: unknown) => ({ error }),
    )
    try {
      await vi.waitFor(() => expect(peak).toBeGreaterThan(1))
    } finally {
      hold.resume()
    }
    const outcome = await pending
    if ('error' in outcome) throw outcome.error
    const { results } = outcome
    expect(results.map(({ key }) => key)).toEqual(await storage.list(''))
    expect(peak).toBeLessThanOrEqual(32)
    expect(active).toBe(0)
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
})
