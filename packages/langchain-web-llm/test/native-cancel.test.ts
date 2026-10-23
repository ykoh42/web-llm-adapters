import { afterEach, expect, it, vi } from 'vitest'
import { getRetryable } from '@langchain/core/errors'
import { BrowserChatWebLLM, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, chunks, collect, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

for (const mode of ['local', 'browser'] as const) {
  it.each(['invoke', 'stream', 'events'] as const)(
    `${mode} %s marks native cancellation as terminal and keeps the engine reusable`,
    async (method) => {
      const { engine, create, model: local } = setup(chunk({ content: 'Recovered' }, 'stop'))
      create.mockResolvedValueOnce(
        chunks(chunk({ content: 'Before cancellation' }), chunk({}, 'abort')),
      )
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmLangChainBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, input) {
          const value = await run(structuredClone(input))
          return value === undefined ? value : JSON.parse(JSON.stringify(value))
        },
      }
      const model =
        mode === 'local' ? local : new BrowserChatWebLLM(page, { modelId: 'test-model' })
      async function run() {
        if (method === 'invoke')
          return model
            .withRetry({
              stopAfterAttempt: 2,
              onFailedAttempt(error) {
                if (getRetryable(error) === false) throw error
              },
            })
            .invoke('Hello')
        if (method === 'stream') return collect(await model.stream('Hello'))
        return collect(model.streamEvents('Hello'))
      }
      try {
        const failure = await run().catch((error: unknown) => error)
        expect(failure).toMatchObject({
          name: 'AbortError',
          message: 'WebLLM cancelled generation.',
        })
        expect(getRetryable(failure)).toBe(false)
        expect(create).toHaveBeenCalledOnce()
        expect((await model.invoke('Try again.')).text).toBe('Recovered')
        expect(create).toHaveBeenCalledTimes(2)
      } finally {
        await bridge.close()
      }
    },
  )
}
