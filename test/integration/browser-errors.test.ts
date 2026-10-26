import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import {
  BrowserModel,
  createBrowserBridge as strandsBridge,
} from '../../packages/strands-web-llm/src/index.js'
import {
  BrowserChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from '../../packages/langchain-web-llm/src/index.js'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

afterEach(() => vi.unstubAllGlobals())

function badCause(error: Error) {
  return Object.defineProperty(error, 'cause', {
    get() {
      throw new Error('Cannot read this cause')
    },
  })
}

const failures = [
  {
    name: 'unreadable value',
    error: () => Object.create(null),
    expected: { name: 'Error', message: 'The request failed with an unreadable error.' },
  },
  {
    name: 'unreadable message',
    error: () =>
      Object.defineProperty(new TypeError('Native failure'), 'message', {
        get() {
          throw new Error('Cannot read this message')
        },
      }),
    expected: { name: 'TypeError', message: 'The request failed with an unreadable error.' },
  },
  {
    name: 'unreadable cause',
    error: () => badCause(new TypeError('Native failure')),
    expected: { name: 'TypeError', message: 'Native failure' },
  },
  {
    name: 'unreadable nested cause',
    error: () =>
      new TypeError('Native failure', { cause: badCause(new RangeError('Original cause')) }),
    expected: {
      name: 'TypeError',
      message: 'Native failure',
      cause: { name: 'RangeError', message: 'Original cause' },
    },
  },
  {
    name: 'unreadable provider fields',
    error: () =>
      new Proxy(new TypeError('Native failure'), {
        ownKeys() {
          throw new Error('Cannot inspect provider fields')
        },
      }),
    expected: { name: 'TypeError', message: 'Native failure' },
  },
]

for (const client of ['strands', 'langchain']) {
  it.each(failures)(
    `${client} transfers $name failures and recovers`,
    async ({ error, expected }) => {
      const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
      create.mockRejectedValueOnce(error())
      const bridge = client === 'strands' ? strandsBridge(engine) : langchainBridge(engine)
      vi.stubGlobal('window', {
        webllmBridge: bridge,
        webllmLangChainBridge: bridge,
      })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      async function invoke() {
        if (client === 'langchain')
          return new BrowserChatWebLLM(page, { modelId: 'test-model' }).invoke('Hello')
        const model = new BrowserModel(page, { modelId: 'test-model' })
        return collect(
          model.stream([new Message({ role: 'user', content: [new TextBlock('Hello')] })]),
        )
      }
      try {
        await expect(invoke()).rejects.toMatchObject(expected)
        expect(JSON.stringify(await invoke())).toContain('Recovered')
        expect(create).toHaveBeenCalledTimes(2)
      } finally {
        await bridge.close()
      }
    },
  )
}
