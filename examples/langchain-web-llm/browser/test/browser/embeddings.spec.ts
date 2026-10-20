import { BrowserWebLLMEmbeddings } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'

test('whitespace embedding inputs keep native values through local and browser adapters', async ({
  page,
}) => {
  const modelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  const documents = [' ', '\t', '\n', '\r\n', '\u00a0', '\u2003', ' \tflower\n'].flatMap((text) => [
    text,
    'A flower.',
  ])
  const queries = [' ', '\t', '\n', ' \u00a0 ']
  await page.evaluate((id) => window.checks.load(id), modelId)
  const expected = await page.evaluate(
    async ({ modelId, documents, queries }) => {
      const engine = window.checks.engine()
      const original = engine.embeddings.create.bind(engine.embeddings)
      const response = await original({
        model: modelId,
        input: documents,
        encoding_format: 'float',
      })
      const expectedDocuments = response.data.map((item) => [...item.embedding])
      const expectedQueries: number[][] = []
      for (const query of queries) {
        const response = await original({
          model: modelId,
          input: ['Represent this sentence for searching relevant passages: ' + query],
          encoding_format: 'float',
        })
        expectedQueries.push([...response.data[0]!.embedding])
      }
      window.embeddingAttempts = []
      engine.embeddings.create = new Proxy(original, {
        apply(target, self, args) {
          const input: unknown = args[0].input
          if (!Array.isArray(input) || !input.every((text) => typeof text === 'string'))
            throw new Error('This test needs string embedding batches.')
          window.embeddingAttempts.push([...input])
          return Reflect.apply(target, self, args)
        },
      })
      return { documents: expectedDocuments, queries: expectedQueries }
    },
    { modelId, documents, queries },
  )
  expect(expected.documents).toHaveLength(documents.length)
  expect(
    [...expected.documents, ...expected.queries].every(
      (vector) => vector.length === 384 && vector.every(Number.isFinite),
    ),
  ).toBe(true)

  const local = await page.evaluate(
    async ({ documents, queries }) => {
      const embeddings = window.checks.embeddings()
      const vectors = await embeddings.embedDocuments(documents)
      const queryVectors: number[][] = []
      for (const query of queries) queryVectors.push(await embeddings.embedQuery(query))
      return { documents: vectors, queries: queryVectors }
    },
    { documents, queries },
  )
  expect(local).toEqual(expected)

  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId })
  expect(await embeddings.embedDocuments(documents)).toEqual(expected.documents)
  for (const [index, query] of queries.entries())
    expect(await embeddings.embedQuery(query)).toEqual(expected.queries[index])
  const requests = [
    documents,
    ...queries.map((query) => [
      'Represent this sentence for searching relevant passages: ' + query,
    ]),
  ]
  expect(await page.evaluate(() => window.embeddingAttempts)).toEqual([...requests, ...requests])
})

test('local embeddings preserve real vectors when engine output arrays are reused', async ({
  page,
}) => {
  const modelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const { expected, actual, afterQuery, calls } = await page.evaluate(async (modelId) => {
    const engine = window.checks.engine()
    const original = engine.embeddings.create.bind(engine.embeddings)
    const documents = Array.from({ length: 33 }, (_, index) => `A flower numbered ${index}.`)
    const native = await original({ model: modelId, input: documents, encoding_format: 'float' })
    const expected = native.data.map((item) => [...item.embedding])
    const buffers: number[][] = []
    let calls = 0
    engine.embeddings.create = async (request) => {
      calls++
      const response = await original(request)
      return {
        ...response,
        data: response.data.map((item) => {
          const buffer = (buffers[item.index] ??= [])
          buffer.splice(0, buffer.length, ...item.embedding)
          return { ...item, embedding: buffer }
        }),
      }
    }
    try {
      const embeddings = window.checks.embeddings()
      const actual = await embeddings.embedDocuments(documents)
      const beforeQuery = actual.map((vector) => [...vector])
      await embeddings.embedQuery('A car.')
      return { expected, actual: beforeQuery, afterQuery: actual, calls }
    } finally {
      engine.embeddings.create = original
    }
  }, modelId)

  expect(calls).toBe(3)
  expect(actual).toEqual(expected)
  expect(afterQuery).toEqual(expected)
})

test('real embedding batches keep native vectors after an injected transient failure', async ({
  page,
}) => {
  const modelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const engine = window.checks.engine()
    const original = engine.embeddings.create.bind(engine.embeddings)
    Object.assign(window, { embeddingAttempts: [] })
    let failed = false
    engine.embeddings.create = new Proxy(original, {
      async apply(target, self, args) {
        const input: unknown = args[0].input
        if (!Array.isArray(input) || !input.every((text) => typeof text === 'string'))
          throw new Error('This test needs string embedding batches.')
        window.embeddingAttempts.push([...input])
        const result = await Reflect.apply(target, self, args)
        if (input[0] === 'A car.' && !failed) {
          failed = true
          throw new TypeError('Failed to fetch', { cause: new SyntaxError('Lost batch response') })
        }
        return result
      },
    })
  })
  const failures: unknown[] = []
  const embeddings = new BrowserWebLLMEmbeddings(page, {
    modelId,
    batchSize: 1,
    maxRetries: 1,
    onFailedAttempt: (error) => {
      failures.push(error)
    },
  })
  const documents = ['A flower.', 'A car.', 'A tree.']
  const vectors = await embeddings.embedDocuments(documents)
  expect(await page.evaluate(() => window.embeddingAttempts)).toEqual([
    ['A flower.'],
    ['A car.'],
    ['A car.'],
    ['A tree.'],
  ])
  expect(failures).toHaveLength(1)
  expect(failures[0]).toBeInstanceOf(TypeError)
  expect(failures[0]).toMatchObject({ message: 'Failed to fetch', cause: expect.any(SyntaxError) })
  const native = await page.evaluate(
    async ({ modelId, documents }) => {
      const vectors: number[][] = []
      for (const text of documents) {
        const result = await window.checks
          .engine()
          .embeddings.create({ model: modelId, input: [text], encoding_format: 'float' })
        vectors.push(result.data[0]!.embedding)
      }
      return vectors
    },
    { modelId, documents },
  )
  expect(vectors).toEqual(native)
  expect(vectors.every((vector) => vector.length === 384 && vector.every(Number.isFinite))).toBe(
    true,
  )
})

test('malformed real embedding output fails without retrying and leaves the engine usable', async ({
  page,
}) => {
  const modelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const engine = window.checks.engine()
    const original = engine.embeddings.create.bind(engine.embeddings)
    Object.assign(window, { embeddingCalls: 0 })
    let first = true
    engine.embeddings.create = new Proxy(original, {
      async apply(target, self, args) {
        window.embeddingCalls++
        const result = await Reflect.apply(target, self, args)
        if (first) {
          first = false
          result.data[0].embedding[0] = Number.NaN
        }
        return result
      },
    })
  })
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId, maxRetries: 1 })
  await expect(embeddings.embedDocuments(['A flower.'])).rejects.toBeInstanceOf(TypeError)
  expect(await page.evaluate(() => window.embeddingCalls)).toBe(1)
  const vector = await embeddings.embedQuery('A flower.')
  expect(vector).toHaveLength(384)
  expect(vector.every(Number.isFinite)).toBe(true)
})

declare global {
  interface Window {
    embeddingAttempts: string[][]
    embeddingCalls: number
  }
}
