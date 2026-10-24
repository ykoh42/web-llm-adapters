import { setImmediate as nextTurn } from 'node:timers/promises'
import { afterEach, expect, it, vi } from 'vitest'
import { browser } from '../fixtures/browser.js'
import { ContextOverflowError, getRetryable } from '@langchain/core/errors'
import { BrowserWebLLMEmbeddings, type BrowserPage } from '../src/index.js'
import { pause } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

it('uses browser embeddings in LangGraph memory and waits for native cleanup', async () => {
  const { InMemoryStore } = await import('@langchain/langgraph')
  const { page, engine, bridge } = browser()
  const evaluate = vi.spyOn(page, 'evaluate')
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => ({
    object: 'list',
    model: 'embedding',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: (request.input as string[]).map((text, index) => ({
      object: 'embedding',
      index,
      embedding: text.includes('orchid') ? [1, 0] : [0, 1],
    })),
  }))
  const embeddings = new BrowserWebLLMEmbeddings(page, {
    modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
  })
  expect(await embeddings.embedQuery('orchid')).toEqual([1, 0])
  expect(evaluate.mock.calls[0]?.[1]).toMatchObject({
    config: { modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4' },
    texts: ['orchid'],
    type: 'query',
  })
  expect(create.mock.calls[0]?.[0].input).toEqual([
    'Represent this sentence for searching relevant passages: orchid',
  ])
  const store = new InMemoryStore({ index: { embeddings, dims: 2 } })
  await store.put(['notes'], 'flower', { text: 'An orchid.' })
  await store.put(['notes'], 'car', { text: 'A car.' })
  expect((await store.search(['notes'], { query: 'orchid', limit: 1 }))[0]?.key).toBe('flower')
  const hold = pause()
  const started = pause()
  create.mockImplementationOnce(async () => {
    started.resume()
    await hold.wait
    return {
      object: 'list',
      model: 'embedding',
      usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      data: [{ object: 'embedding', index: 0, embedding: [1, 0] }],
    }
  })
  const pending = embeddings.embedQuery('orchid')
  await started.wait
  const closing = bridge.close()
  const completed = closing.then(() => true)
  expect(await Promise.race([completed, nextTurn().then(() => false)])).toBe(false)
  hold.resume()
  await pending
  await completed
  await expect(embeddings.embedQuery('orchid')).rejects.toThrow('closed')
})

it('retries only the failed browser embedding batch', async () => {
  const { page, engine, bridge } = browser()
  const attempts: string[][] = []
  let failed = false
  vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => {
    const texts = request.input as string[]
    attempts.push([...texts])
    if (texts[0] === 'Two' && !failed) {
      failed = true
      throw new Error('Temporary failure')
    }
    return {
      object: 'list',
      model: 'embedding-model',
      usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      data: texts.map((text, index) => ({
        object: 'embedding',
        index,
        embedding: [text.length, 1],
      })),
    }
  })
  const model = new BrowserWebLLMEmbeddings(page, {
    modelId: 'embedding-model',
    batchSize: 1,
    maxRetries: 1,
  })
  expect(await model.embedDocuments(['One', 'Two', 'Three'])).toEqual([
    [3, 1],
    [3, 1],
    [5, 1],
  ])
  expect(attempts).toEqual([['One'], ['Two'], ['Two'], ['Three']])
  await bridge.close()
})

it('preserves an embedding context error across a real JSON error boundary without retrying', async () => {
  const { engine, bridge } = browser()
  const native = Object.assign(new Error('The input is too long.'), {
    name: 'EmbeddingExceedContextWindowSizeError',
  })
  const create = vi.spyOn(engine.embeddings, 'create').mockRejectedValue(native)
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const result = await Promise.resolve()
        .then(() => run(structuredClone(arg)))
        .then(
          (value) => ({ value }),
          (error: unknown) => ({
            error: {
              name: error instanceof Error ? error.name : 'Error',
              message: error instanceof Error ? error.message : String(error),
            },
          }),
        )
      if ('error' in result)
        throw Object.assign(new Error(result.error.message), { name: result.error.name })
      return JSON.parse(JSON.stringify(result.value))
    },
  }
  const model = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', maxRetries: 1 })
  const failure = await model.embedQuery('Orchid').catch((error: unknown) => error)
  expect(ContextOverflowError.isInstance(failure)).toBe(true)
  expect(getRetryable(failure)).toBe(false)
  expect(failure).toMatchObject({ cause: { name: native.name, message: native.message } })
  expect(create).toHaveBeenCalledOnce()
  await bridge.close()
})

it('does not retry a malformed browser embedding response', async () => {
  const { page, engine, bridge } = browser()
  const create = vi.spyOn(engine.embeddings, 'create').mockResolvedValue({
    object: 'list',
    model: 'embedding-model',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: [{ object: 'embedding', index: 0, embedding: [Number.NaN] }],
  })
  const model = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', maxRetries: 1 })
  const failure = await model.embedDocuments(['One']).catch((error: unknown) => error)
  expect(create).toHaveBeenCalledOnce()
  expect(failure).toBeInstanceOf(TypeError)
  await bridge.close()
})

it('checks vector dimensions across browser embedding batches', async () => {
  const { page, engine, bridge } = browser()
  let call = 0
  vi.spyOn(engine.embeddings, 'create').mockImplementation(async () => ({
    object: 'list',
    model: 'embedding-model',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: [{ object: 'embedding', index: 0, embedding: call++ === 0 ? [1, 0] : [1] }],
  }))
  const model = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', batchSize: 1 })
  await expect(model.embedDocuments(['One', 'Two'])).rejects.toThrow('matching dimensions')
  await bridge.close()
})

it('bounds browser embedding requests with the configured batch size', async () => {
  const { page, engine, bridge } = browser()
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => {
    const documents = request.input as string[]
    return {
      object: 'list',
      model: 'embedding-model',
      usage: {
        prompt_tokens: documents.length,
        total_tokens: documents.length,
        extra: { prefill_tokens_per_s: 1 },
      },
      data: documents.map((text, index) => ({
        object: 'embedding',
        index,
        embedding: [Number(text.slice(8)), 1],
      })),
    }
  })
  const embeddings = new BrowserWebLLMEmbeddings(page, {
    modelId: 'embedding-model',
    batchSize: 2,
  })
  const documents = Array.from({ length: 5 }, (_, index) => `Document ${index}`)

  const result = await embeddings.embedDocuments(documents)

  expect(create.mock.calls.map(([request]) => request.input)).toEqual([
    ['Document 0', 'Document 1'],
    ['Document 2', 'Document 3'],
    ['Document 4'],
  ])
  expect(result).toEqual(documents.map((_, index) => [index, 1]))
  await bridge.close()
})

it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
  'rejects invalid browser embedding batch size %s',
  async (batchSize) => {
    const { page } = browser()

    expect(
      () => new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', batchSize }),
    ).toThrow('batchSize must be a positive integer.')
  },
)

it('explains how to expose the browser bridge when embedding setup is missing', async () => {
  const { page } = browser()
  vi.stubGlobal('window', {})
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })
  await expect(embeddings.embedQuery('hello')).rejects.toThrow(
    'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserWebLLMEmbeddings.',
  )
})

it('handles empty embedding inputs before crossing the browser boundary', async () => {
  const { page, bridge } = browser()
  const evaluate = vi.spyOn(page, 'evaluate')
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })

  await expect(embeddings.embedDocuments([])).resolves.toEqual([])
  await expect(embeddings.embedDocuments([''])).rejects.toThrow('WebLLM cannot embed empty text.')
  await expect(embeddings.embedQuery('')).rejects.toThrow('WebLLM cannot embed empty text.')
  expect(evaluate).not.toHaveBeenCalled()
  await bridge.close()
})

it('preserves embedding whitespace across the browser boundary', async () => {
  const { page, bridge, engine } = browser()
  const create = vi.spyOn(engine.embeddings, 'create').mockResolvedValue({
    object: 'list',
    model: 'embedding-model',
    data: [{ object: 'embedding', index: 0, embedding: [1, 0] }],
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
  })
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })
  try {
    for (const text of [' ', '\t', '\n', '\r\n', '\u00a0', '\u2003', ' \tflower\n']) {
      await expect(embeddings.embedDocuments([text])).resolves.toEqual([[1, 0]])
      await expect(embeddings.embedQuery(text)).resolves.toEqual([1, 0])
      expect(create.mock.calls.at(-2)?.[0].input).toEqual([text])
      expect(create.mock.calls.at(-1)?.[0].input).toEqual([text])
    }
  } finally {
    await bridge.close()
  }
})

it('snapshots browser embedding inputs before a delayed page transfer', async () => {
  const { engine, bridge } = browser()
  const hold = pause()
  const page: BrowserPage = {
    async evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T> {
      await hold.wait
      const result = await run(structuredClone(arg))
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => ({
    object: 'list',
    model: 'embedding-model',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: (request.input as string[]).map((text, index) => ({
      object: 'embedding',
      index,
      embedding: text === 'Before' ? [1, 0] : [0, 1],
    })),
  }))
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })
  const documents = ['Before']
  const pending = embeddings.embedDocuments(documents)
  documents[0] = 'After'
  hold.resume()

  await expect(pending).resolves.toEqual([[1, 0]])
  expect(create.mock.calls[0]?.[0].input).toEqual(['Before'])
  await bridge.close()
})
