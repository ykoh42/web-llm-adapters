import { afterEach, expect, it, vi } from 'vitest'
import { HumanMessage } from '@langchain/core/messages'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

for (const browser of [false, true]) {
  it(`preserves native response identity and model metadata (browser: ${browser})`, async () => {
    const native = [
      { ...chunk({ content: 'Hello' }), id: 'native-response', model: 'served-model' },
      { ...chunk({}, 'stop'), id: 'native-response', model: 'served-model' },
    ]
    const { engine } = setup(...native)
    const bridge = createBrowserBridge(engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        const result = await run(structuredClone(arg))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const onChunk = vi.fn()
    const options = { modelId: 'test-model', onChunk }
    const model = browser
      ? new BrowserChatWebLLM(page, options)
      : new ChatWebLLM({ engine, ...options })
    try {
      const result = await model.invoke('Hello')
      expect(result.id).toBe('native-response')
      expect(result.text).toBe('Hello')
      expect(result.response_metadata).toMatchObject({
        model: 'served-model',
        model_name: 'served-model',
        finish_reason: 'stop',
      })
      expect(onChunk.mock.calls.map(([value]) => value)).toEqual(native)
      const events = []
      for await (const event of model._streamChatModelEvents([new HumanMessage('Hello')], {}))
        events.push(event)
      expect(events[0]).toEqual({ event: 'message-start', id: 'native-response' })
    } finally {
      await bridge.close()
    }
  })
}
