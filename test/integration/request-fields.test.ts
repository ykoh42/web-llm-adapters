import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge as createStrandsBridge,
} from 'strands-web-llm'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

const fields = [
  { key: 'model', value: 'other-model' },
  { key: 'messages', value: [{ role: 'user', content: 'Replaced' }] },
  { key: 'prompt', value: 'Replaced' },
  { key: 'tools', value: [{ type: 'function', function: { name: 'unregistered' } }] },
  { key: 'tool_choice', value: 'auto' },
  { key: 'n', value: 2 },
  { key: 'stream', value: false },
  { key: 'stream_options', value: { include_usage: false } },
  { key: 'tools', value: undefined },
  { key: 'n', value: undefined },
]
const clients = ['langchain', 'langchain-browser', 'pi', 'strands', 'strands-browser'] as const

afterEach(() => vi.unstubAllGlobals())

for (const client of clients) {
  it.each(fields.flatMap((field) => ['defaults', 'call'].map((scope) => ({ ...field, scope }))))(
    `${client} rejects request.$key from $scope before native inference`,
    async ({ key, value, scope }) => {
      const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
      const bridge = createBrowserBridge(engine)
      const strandsBridge = createStrandsBridge(engine)
      vi.stubGlobal('window', { webllmLangChainBridge: bridge, webllmBridge: strandsBridge })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      // Dynamic keys represent JavaScript callers and decoded configuration files.
      const request = { seed: 7, [key]: value }
      const onChunk = vi.fn()
      const options = {
        modelId: 'test-model',
        request: scope === 'defaults' ? request : { seed: 7 },
        onChunk,
      }
      async function invoke() {
        if (client === 'pi') {
          const provider = createWebLLMProvider({ engine, contextWindow: 8192, ...options })
          const result = await provider
            .stream(
              provider.getModels()[0]!,
              normalizeContext({ messages: [{ role: 'user', content: 'Hello', timestamp: 0 }] }),
              scope === 'call' ? { request } : {},
            )
            .result()
          expect(result.stopReason).toBe('error')
          expect(result.errorMessage).toBe(`Use the provider API instead of overriding ${key}.`)
          return
        }
        if (client === 'strands' || client === 'strands-browser') {
          const model =
            client === 'strands'
              ? new WebLLMModel({ engine, ...options })
              : new BrowserModel(page, options)
          if (scope === 'call') model.updateConfig({ request })
          await collect(
            model.stream([new Message({ role: 'user', content: [new TextBlock('Hello')] })]),
          )
          return
        }
        const model =
          client === 'langchain'
            ? new ChatWebLLM({ engine, ...options })
            : new BrowserChatWebLLM(page, options)
        await model.invoke('Hello', scope === 'call' ? { request } : {})
      }
      try {
        if (client === 'pi') await invoke()
        else
          await expect(invoke()).rejects.toMatchObject({
            name: 'Error',
            message: `Use the provider API instead of overriding ${key}.`,
          })
        expect(create).not.toHaveBeenCalled()
        expect(onChunk).not.toHaveBeenCalled()
        expect((await new ChatWebLLM({ engine, modelId: 'test-model' }).invoke('Next')).text).toBe(
          'Recovered',
        )
        expect(create).toHaveBeenCalledOnce()
      } finally {
        await Promise.all([bridge.close(), strandsBridge.close()])
      }
    },
  )
}

it.each([false, true])(
  'keeps Strands config after a rejected update (browser: %s)',
  async (browser) => {
    const { engine } = setupEngine()
    const page: BrowserPage = { evaluate: vi.fn() }
    const options = { modelId: 'test-model', request: { seed: 7, tool_choice: 'none' as const } }
    const model = browser
      ? new BrowserModel(page, options)
      : new WebLLMModel({ engine, ...options })
    const previous = model.getConfig()
    const request = { seed: 99, tools: [] }

    expect(() => model.updateConfig({ request })).toThrow('overriding tools')
    expect(model.getConfig()).toEqual(previous)
    model.updateConfig({ request: { seed: 99, tool_choice: 'none' } })
    expect(model.getConfig().request).toEqual({ seed: 99, tool_choice: 'none' })
  },
)

for (const browser of [false, true]) {
  it.each(['none', undefined] as const)(
    `keeps Strands optional tool_choice %s (browser: ${browser})`,
    async (tool_choice) => {
      const { engine, create } = setupEngine(chunk({ content: 'Done' }, 'stop'))
      const bridge = createStrandsBridge(engine)
      vi.stubGlobal('window', { webllmBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const request = { seed: 7 }
      Object.assign(request, { tool_choice })
      const options = { modelId: 'test-model', request }
      try {
        const model = browser
          ? new BrowserModel(page, options)
          : new WebLLMModel({ engine, ...options })
        const output = await collect(
          model.stream([new Message({ role: 'user', content: [new TextBlock('Hello')] })]),
        )
        expect(output).toContainEqual({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: 'Done' },
        })
        expect(create).toHaveBeenCalledOnce()
        expect(create.mock.calls[0]?.[0].tool_choice).toBe(tool_choice)
      } finally {
        await bridge.close()
      }
    },
  )
}
