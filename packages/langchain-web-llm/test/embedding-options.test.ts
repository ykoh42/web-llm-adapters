import { afterEach, expect, it, vi } from 'vitest'
import { stampRetryable, getRetryable } from '@langchain/core/errors'
import { BrowserWebLLMEmbeddings, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

it.each([
  { custom: false, marked: false },
  { custom: true, marked: false },
  { custom: true, marked: undefined },
])(
  'preserves embedding retry metadata (custom=$custom, marked=$marked)',
  async ({ custom, marked }) => {
    const { engine } = setup()
    const native = Object.assign(new Error('GPU busy'), {
      status: 429,
      code: 'GPU_BUSY',
      retryAfterMs: 50,
    })
    if (marked !== undefined) stampRetryable(native, marked)
    const create = vi.spyOn(engine.embeddings, 'create').mockRejectedValue(native)
    const bridge = createBrowserBridge(engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        return JSON.parse(JSON.stringify(await run(structuredClone(arg))))
      },
    }
    const stopped = new Error('Do not retry')
    const onFailedAttempt = vi.fn((error: unknown) => {
      expect(error).toMatchObject({
        name: 'Error',
        message: 'GPU busy',
        status: 429,
        code: 'GPU_BUSY',
        retryAfterMs: 50,
      })
      expect(getRetryable(error)).toBe(marked)
      throw stopped
    })
    const model = new BrowserWebLLMEmbeddings(page, {
      modelId: 'embedding-model',
      maxRetries: 1,
      ...(custom && { onFailedAttempt }),
    })
    try {
      const failure = await model.embedQuery('Hello').catch((error: unknown) => error)
      if (custom) {
        expect(failure).toBe(stopped)
        expect(onFailedAttempt).toHaveBeenCalledOnce()
      } else {
        expect(getRetryable(failure)).toBe(false)
        expect(failure).toMatchObject({ message: 'GPU busy', status: 429, code: 'GPU_BUSY' })
      }
      expect(create).toHaveBeenCalledOnce()
      create.mockResolvedValue({
        object: 'list',
        model: 'embedding-model',
        data: [{ object: 'embedding', index: 0, embedding: [1, 2] }],
        usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      })
      expect(await model.embedQuery('Recover')).toEqual([1, 2])
    } finally {
      await bridge.close()
    }
  },
)

it('respects failed-attempt options on a direct in-page embedding call', async () => {
  const { engine } = setup()
  const native = new Error('GPU busy')
  const stopped = new Error('Do not retry')
  const create = vi.spyOn(engine.embeddings, 'create').mockRejectedValue(native)
  const onFailedAttempt = vi.fn((error: unknown) => {
    expect(error).toBe(native)
    throw stopped
  })
  const bridge = createBrowserBridge(engine)
  try {
    expect(
      await bridge.embed({
        config: { modelId: 'embedding-model', maxRetries: 1, onFailedAttempt },
        texts: ['Hello'],
        type: 'query',
      }),
    ).toMatchObject({ error: { message: stopped.message } })
    expect(create).toHaveBeenCalledOnce()
    expect(onFailedAttempt).toHaveBeenCalledOnce()
  } finally {
    await bridge.close()
  }
})
