import { afterEach, describe, expect, it, vi } from 'vitest'
import { BaseCallbackHandler } from '@langchain/core/callbacks/base'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, pause, setup } from '../fixtures/engine.js'
import { browser } from '../fixtures/browser.js'

afterEach(() => vi.unstubAllGlobals())

it('notifies an enabled browser callback before reading the rest of the response', async () => {
  const fixture = browser()
  const hold = pause()
  fixture.create.mockResolvedValueOnce(
    (async function* () {
      yield chunk({ content: 'First' })
      await hold.wait
      yield chunk({ content: 'Second' }, 'stop')
    })(),
  )
  const onToken = vi.fn(() => hold.resume())
  const handler = BaseCallbackHandler.fromMethods({ handleLLMNewToken: onToken })
  handler.ignoreLLM = true
  handler.awaitHandlers = true
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const result = await fixture.page.evaluate(run, arg)
      if (typeof arg === 'string') handler.ignoreLLM = false
      return result
    },
  }
  const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
  const pending = model.invoke('Hi', { callbacks: [handler] })
  try {
    await vi.waitFor(() => expect(onToken).toHaveBeenCalled())
    expect((await pending).text).toBe('FirstSecond')
  } finally {
    hold.resume()
    await pending.catch(() => {})
    await fixture.bridge.close()
  }
})

describe.each(['local', 'browser'])('%s token callbacks', (mode) => {
  it('honors a registered handler enabled after the native stream starts', async () => {
    const { engine, page, bridge } = browser(
      chunk({ content: 'First' }),
      chunk({ content: 'Second' }),
      chunk({}, 'stop'),
    )
    const onToken = vi.fn<(token: string) => void>()
    const handler = BaseCallbackHandler.fromMethods({ handleLLMNewToken: onToken })
    handler.ignoreLLM = true
    handler.awaitHandlers = true
    const options = {
      modelId: 'test-model',
      onChunk: () => {
        handler.ignoreLLM = false
      },
    }
    const model =
      mode === 'local'
        ? new ChatWebLLM({ engine, ...options })
        : new BrowserChatWebLLM(page, options)
    try {
      const result = await model.invoke('Hi', { callbacks: [handler] })

      expect(result.text).toBe('FirstSecond')
      expect(onToken.mock.calls.map(([token]) => token).join('')).toBe(result.text)
    } finally {
      await bridge.close()
    }
  })

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
