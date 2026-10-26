import { afterEach, describe, expect, it, vi } from 'vitest'
import { HumanMessage } from '@langchain/core/messages'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, collect, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

describe.each(['local', 'browser'])('%s callback errors', (mode) => {
  it.each(
    ['onChunk', 'onFailedAttempt'].flatMap((callback) =>
      [undefined, null, 0, 'Observer failed', new TypeError('Observer failed')].map((error) => ({
        callback,
        error,
      })),
    ),
  )('preserves $callback failure and recovers: $error', async ({ callback, error }) => {
    const { engine, create, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const bridge = createBrowserBridge(engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, input) {
        const result = await run(structuredClone(input))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const handler = vi.fn().mockImplementationOnce(() => {
      throw error
    })
    const options = { modelId: 'test-model', maxRetries: 1, [callback]: handler }
    const model =
      mode === 'local'
        ? new ChatWebLLM({ engine, ...options })
        : new BrowserChatWebLLM(page, options)
    if (callback === 'onFailedAttempt') create.mockRejectedValueOnce(new Error('Opening failed'))
    try {
      const failure: unknown = await model.invoke('Hi').catch((error: unknown) => error)
      if (error instanceof Error) expect(failure).toBe(error)
      else {
        expect(failure).toBeInstanceOf(Error)
        expect(failure).toHaveProperty('message', String(error))
        expect(failure).toHaveProperty('cause', error)
      }
      expect(create).toHaveBeenCalledOnce()
      expect(stop).toHaveBeenCalledTimes(mode === 'browser' || callback === 'onChunk' ? 1 : 0)
      expect((await model.invoke('Next')).text).toBe('Recovered')
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      await bridge.close()
    }
  })

  it('awaits async chunk callbacks and recovers after rejection', async () => {
    const { engine, create, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const bridge = createBrowserBridge(engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, input) {
        const result = await run(structuredClone(input))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const onChunk = vi.fn().mockImplementationOnce(async () => {
      await Promise.resolve()
      throw new Error('Async observer failed')
    })
    const options = { modelId: 'test-model', maxRetries: 1, onChunk }
    const model =
      mode === 'local'
        ? new ChatWebLLM({ engine, ...options })
        : new BrowserChatWebLLM(page, options)
    try {
      await expect(model.invoke('Hi')).rejects.toThrow('Async observer failed')
      expect(onChunk).toHaveBeenCalledOnce()
      expect(stop).toHaveBeenCalledOnce()
      expect((await model.invoke('Next')).text).toBe('Recovered')
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      await bridge.close()
    }
  })
})

it.each([null, 0, 'Cancelled by caller'])(
  'keeps a browser cancellation reason: %s',
  async (reason) => {
    const { engine } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const bridge = createBrowserBridge(engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, input) {
        const result = await run(structuredClone(input))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const controller = new AbortController()
    const onChunk = vi.fn().mockImplementationOnce(() => controller.abort(reason))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model', onChunk })
    try {
      await expect(
        collect(
          model._streamResponseChunks([new HumanMessage('Hi')], { signal: controller.signal }),
        ),
      ).rejects.toBe(reason)
      expect((await model.invoke('Next')).text).toBe('Recovered')
    } finally {
      await bridge.close()
    }
  },
)
