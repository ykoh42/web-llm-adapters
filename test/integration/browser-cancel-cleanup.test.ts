import { afterEach, expect, it, vi } from 'vitest'
import { BrowserModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { chunk, collect, pause, setupEngine } from '../../fixtures/web-llm.js'
import { startBrowserCall } from '../../shared/browser/start.js'

afterEach(() => vi.unstubAllGlobals())

it('cleans up a browser call when cancellation precedes an already completed start', async () => {
  const controller = new AbortController()
  const stop = vi.fn(async () => {})
  controller.abort()

  await expect(startBrowserCall(Promise.resolve('call'), stop, controller.signal)).rejects.toBe(
    controller.signal.reason,
  )
  expect(stop).toHaveBeenCalledExactlyOnceWith('call')
})

it.each(['strands', 'langchain'] as const)(
  '%s waits for a cancelled native read when closing the bridge',
  async (client) => {
    const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
    const reading = pause()
    const release = pause()
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          reading.resume()
          await release.wait
          return { done: true as const, value: undefined }
        },
        return: async () => ({ done: true as const, value: undefined }),
      }),
    })
    const bridge = client === 'strands' ? strandsBridge(engine) : langchainBridge(engine)
    const stop = vi.spyOn(bridge, 'stop')
    vi.stubGlobal('window', { webllmBridge: bridge, webllmLangChainBridge: bridge })
    const page: BrowserPage = { evaluate: async (run, arg) => run(arg) }
    const controller = new AbortController()
    const pending =
      client === 'strands'
        ? collect(
            new BrowserModel(page, { modelId: 'test-model' }).stream([], {
              cancelSignal: controller.signal,
            }),
          )
        : new BrowserChatWebLLM(page, { modelId: 'test-model' }).invoke('Hello', {
            signal: controller.signal,
          })
    void pending.catch(() => {})
    await reading.wait
    controller.abort()
    await pending.catch(() => {})
    await vi.waitFor(() => expect(stop).toHaveBeenCalledOnce())
    let stopped = false
    const stopping = bridge.stop(stop.mock.calls[0]![0]).then(() => {
      stopped = true
    })
    let closed = false
    const closing = bridge.close().then(() => {
      closed = true
    })
    try {
      await new Promise((resolve) => setTimeout(resolve, 10))
      expect(stopped).toBe(false)
      expect(closed).toBe(false)
    } finally {
      release.resume()
      await Promise.all([stopping, closing])
    }
  },
)
