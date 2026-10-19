import { describe, expect, it, vi } from 'vitest'
import { InMemoryStore } from '@langchain/langgraph'
import { WebLLMEmbeddings } from '../src/index.js'
import { setup } from '../fixtures/engine.js'

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
function vectors(values: number[][]) {
  const { engine } = setup()
  const create = vi.spyOn(engine.embeddings, 'create').mockResolvedValue(embedding(values))
  return {
    create,
    model: new WebLLMEmbeddings({ engine, modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4' }),
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
  it('handles empty batches and rejects empty documents', async () => {
    const { model, create } = vectors([])
    expect(await model.embedDocuments([])).toEqual([])
    expect(create).not.toHaveBeenCalled()
    await expect(model.embedDocuments([' '])).rejects.toThrow('empty')
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
