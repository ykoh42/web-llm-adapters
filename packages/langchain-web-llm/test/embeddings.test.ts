import { describe, expect, it, vi } from 'vitest'
import { InMemoryStore } from '@langchain/langgraph'
import { WebLLMEmbeddings, withEngine } from '../src/index.js'
import { pause, setup } from '../fixtures/engine.js'

function embedding(values: number[][]) {
  return {
    data: values.map((vector, index) => ({
      object: 'embedding' as const,
      index,
      embedding: vector,
    })),
    model: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
    object: 'list' as const,
    usage: { prompt_tokens: 2, total_tokens: 2, extra: { prefill_tokens_per_s: 1 } },
  }
}
function vectors(values: number[][], batchSize?: number) {
  const { engine } = setup()
  const create = vi.spyOn(engine.embeddings, 'create').mockResolvedValue(embedding(values))
  return {
    create,
    model: new WebLLMEmbeddings({
      engine,
      modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
      ...(batchSize !== undefined && { batchSize }),
    }),
  }
}

describe('LangChain embeddings', () => {
  it('applies the recommended Snowflake prefix only to queries', async () => {
    const { model, create } = vectors([[1, 0]])
    expect(await model.embedQuery('Orchid')).toEqual([1, 0])
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      input: ['Represent this sentence for searching relevant passages: Orchid'],
      encoding_format: 'float',
    })
    await model.embedDocuments(['Orchid'])
    expect(create.mock.calls[1]?.[0].input).toEqual(['Orchid'])
  })
  it('supports LangGraph semantic memory through its standard embeddings interface', async () => {
    const { model, create } = vectors([[1, 0]])
    create.mockImplementation(async (request) =>
      embedding(
        (request.input as string[]).map((text) => (text.includes('orchid') ? [1, 0] : [0, 1])),
      ),
    )
    const store = new InMemoryStore({ index: { embeddings: model, dims: 2 } })
    await store.put(['memory'], 'flower', { text: 'An orchid is a flower.' })
    await store.put(['memory'], 'car', { text: 'A car has wheels.' })
    const result = await store.search(['memory'], { query: 'orchid', limit: 1 })
    expect(result[0]?.key).toBe('flower')
  })
  it('restores native output order without altering vectors', async () => {
    const { model, create } = vectors([
      [1, 2],
      [3, 4],
    ])
    create.mockResolvedValueOnce({
      ...embedding([
        [1, 2],
        [3, 4],
      ]),
      data: embedding([
        [1, 2],
        [3, 4],
      ]).data.reverse(),
    })
    expect(await model.embedDocuments(['One', 'Two'])).toEqual([
      [1, 2],
      [3, 4],
    ])
  })
  it('batches document requests by default', async () => {
    const { model, create } = vectors([])
    create.mockImplementation(async (request) => {
      const documents = request.input as string[]
      return embedding(documents.map((text) => [Number(text.slice(8)), 1]))
    })
    const documents = Array.from({ length: 33 }, (_, index) => `Document ${index}`)

    const result = await model.embedDocuments(documents)

    expect(create.mock.calls.map(([request]) => (request.input as string[]).length)).toEqual([
      32, 1,
    ])
    expect(result).toEqual(documents.map((_, index) => [index, 1]))
  })
  it('bounds requests and restores out-of-order indices across batches', async () => {
    const { model, create } = vectors([], 2)
    create.mockImplementation(async (request) => {
      const documents = request.input as string[]
      return {
        ...embedding([]),
        data: documents
          .map((text, index) => ({
            object: 'embedding' as const,
            index,
            embedding: [Number(text.slice(8)), 1],
          }))
          .reverse()
          .map((item, index, items) => ({ ...item, index: items.length - index - 1 })),
      }
    })
    const documents = Array.from({ length: 5 }, (_, index) => `Document ${index}`)

    const result = await model.embedDocuments(documents)

    expect(create.mock.calls.map(([request]) => request.input)).toEqual([
      ['Document 0', 'Document 1'],
      ['Document 2', 'Document 3'],
      ['Document 4'],
    ])
    expect(result).toEqual(documents.map((_, index) => [index, 1]))
  })
  it('checks vector dimensions across document batches', async () => {
    const { model, create } = vectors([], 1)
    create.mockResolvedValueOnce(embedding([[1, 0]])).mockResolvedValueOnce(embedding([[1]]))

    await expect(model.embedDocuments(['First', 'Second'])).rejects.toThrow('matching dimensions')
  })
  it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects invalid batch size %s',
    (batchSize) => {
      const { engine } = setup()
      expect(() => new WebLLMEmbeddings({ engine, modelId: 'embedding-model', batchSize })).toThrow(
        'positive integer',
      )
    },
  )
  it('snapshots documents before waiting for its engine turn', async () => {
    const { model, create } = vectors([[1, 0]])
    const hold = pause()
    const active = withEngine(model.engine, async () => hold.wait)
    const documents = ['Before']
    const pending = model.embedDocuments(documents)
    documents[0] = 'After'
    hold.resume()

    await active
    await pending

    expect(create.mock.calls[0]?.[0].input).toEqual(['Before'])
  })
  it('handles empty batches and rejects empty documents', async () => {
    const { model, create } = vectors([])
    expect(await model.embedDocuments([])).toEqual([])
    expect(create).not.toHaveBeenCalled()
    await expect(model.embedDocuments([' '])).rejects.toThrow('empty')
    await expect(model.embedQuery(' \t')).rejects.toThrow('empty')
    expect(create).not.toHaveBeenCalled()
  })
  it('keeps local retries off when maxRetries is undefined', async () => {
    const { engine } = setup()
    const create = vi.spyOn(engine.embeddings, 'create').mockRejectedValue(new Error('GPU lost'))
    const options = { engine, modelId: 'embedding-model' }
    Object.assign(options, { maxRetries: undefined })
    const model = new WebLLMEmbeddings(options)

    await expect(model.embedQuery('Orchid')).rejects.toThrow('GPU lost')
    expect(create).toHaveBeenCalledOnce()
  })
  it.each([[[NaN]], [[Infinity]], [[]], [[1], [1, 2]]].map((values) => ({ values })))(
    'rejects invalid native vectors $values',
    async ({ values }) => {
      const { model } = vectors(values)
      await expect(model.embedDocuments(values.map(() => 'Text'))).rejects.toThrow(
        /finite|dimensions/,
      )
    },
  )
  it('rejects missing and duplicate indices', async () => {
    const { model, create } = vectors([[1]])
    await expect(model.embedDocuments(['One', 'Two'])).rejects.toThrow('number')
    create.mockResolvedValueOnce({
      ...embedding([[1], [2]]),
      data: [
        { object: 'embedding', index: 0, embedding: [1] },
        { object: 'embedding', index: 0, embedding: [2] },
      ],
    })
    await expect(model.embedDocuments(['One', 'Two'])).rejects.toThrow('indices')
  })
})
