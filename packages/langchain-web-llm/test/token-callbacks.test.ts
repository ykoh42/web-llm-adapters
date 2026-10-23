import { afterEach, describe, expect, it, vi } from 'vitest'
import { BaseCallbackHandler } from '@langchain/core/callbacks/base'
import { BrowserChatWebLLM, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, pause, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

describe.each(['local', 'browser'])('%s token callbacks', (mode) => {
  it.each([
    ['invoke', 'finish'],
    ['invoke', 'cancel'],
    ['stream', 'finish'],
    ['stream', 'cancel'],
  ])('%s lets the caller %s a pending callback', async (method, action) => {
    const local = setup(chunk({ content: 'Answer' }, 'stop'))
    const bridge = createBrowserBridge(local.engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, input) {
        const result = await run(structuredClone(input))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const model =
      mode === 'local' ? local.model : new BrowserChatWebLLM(page, { modelId: 'test-model' })
    const hold = pause()
    const onToken = vi.fn<(token: string) => Promise<void>>(() => hold.wait)
    const handler = BaseCallbackHandler.fromMethods({ handleLLMNewToken: onToken })
    handler.awaitHandlers = true
    const controller = new AbortController()
    const options = { signal: controller.signal, callbacks: [handler] }
    const pending =
      method === 'invoke'
        ? model.invoke('Hi', options)
        : (async () => {
            const text: string[] = []
            for await (const part of await model.stream('Hi', options)) text.push(part.text)
            return { text: text.join('') }
          })()
    let settled = false
    void pending.then(
      () => (settled = true),
      () => (settled = true),
    )
    try {
      await vi.waitFor(() => expect(onToken).toHaveBeenCalledOnce())
      expect(settled).toBe(false)
      if (action === 'cancel') {
        controller.abort()
        await vi.waitFor(() => expect(settled).toBe(true))
        await expect(pending).rejects.toHaveProperty('name', 'AbortError')
      } else {
        hold.resume()
        expect((await pending).text).toBe('Answer')
      }
      expect((await model.invoke('Next')).text).toBe('Answer')
      expect(onToken.mock.calls.map(([token]) => token).join('')).toBe('Answer')
      if (action === 'cancel') expect(onToken).toHaveBeenCalledOnce()
    } finally {
      hold.resume()
      await pending.catch(() => {})
      await bridge.close()
    }
  })
})
