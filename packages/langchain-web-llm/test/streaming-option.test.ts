import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserBridge,
  type BrowserPage,
  type WebLLMModelOptions,
} from '../src/index.js'
import { chunk, collect, setup } from '../fixtures/engine.js'

const bridges: BrowserBridge[] = []
afterEach(async () => {
  try {
    await Promise.all(bridges.splice(0).map((bridge) => bridge.close()))
  } finally {
    vi.unstubAllGlobals()
  }
})

describe.each(['local', 'browser'])('%s streaming option', (transport) => {
  it.each([undefined, false, true])(
    'honors disableStreaming=%s without changing model requests or completed output',
    async (disableStreaming) => {
      const { engine, create } = setup(
        chunk({ content: 'Hello ' }),
        chunk({ content: '세계 <>&' }),
        chunk({}, 'stop'),
      )
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      if (transport === 'browser') {
        const bridge = createBrowserBridge(engine)
        bridges.push(bridge)
        vi.stubGlobal('window', { webllmLangChainBridge: bridge })
      }
      function model(options: Partial<WebLLMModelOptions> = {}) {
        const config = {
          modelId: 'test-model',
          temperature: 0.25,
          maxTokens: 64,
          topP: 0.9,
          stop: ['[END]'],
          ...options,
        }
        return transport === 'browser'
          ? new BrowserChatWebLLM(page, config)
          : new ChatWebLLM({ engine, ...config })
      }
      const input = 'Keep this message: 한글 <>&'
      const options = { request: { seed: 7 } }
      const expected = await model().invoke(input, options)
      const normal = await collect(await model().stream(input, options))
      const selected = model(disableStreaming === undefined ? {} : { disableStreaming })
      const output = await collect(await selected.stream(input, options))

      expect(normal.length).toBeGreaterThan(1)
      expect(output).toHaveLength(disableStreaming ? 1 : normal.length)
      expect(output.map((message) => message.text).join('')).toBe(expected.text)
      if (disableStreaming) {
        expect(output[0]?.content).toEqual(expected.content)
        expect(output[0]?.tool_calls).toEqual(expected.tool_calls)
        expect(output[0]?.response_metadata).toEqual(expected.response_metadata)
      }
      expect(create).toHaveBeenCalledTimes(3)
      for (const [request] of create.mock.calls) expect(request).toEqual(create.mock.calls[0]?.[0])
      expect(create.mock.calls[0]?.[0]).toMatchObject({
        messages: [{ role: 'user', content: input }],
        temperature: 0.25,
        max_tokens: 64,
        top_p: 0.9,
        stop: ['[END]'],
        seed: 7,
      })
    },
  )
})
