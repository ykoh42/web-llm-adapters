import { afterEach, expect, it, vi } from 'vitest'
import { Agent, FunctionTool, ModelError } from '@strands-agents/sdk'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { BrowserModel, WebLLMModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { chunk, chunks, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

const clients = (['Strands', 'LangChain'] as const).flatMap((framework) =>
  [false, true].map((browser) => ({ framework, browser })),
)

function setupModels(browser: boolean) {
  const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
  const strands = strandsBridge(engine)
  const langchain = langchainBridge(engine)
  vi.stubGlobal('window', { webllmBridge: strands, webllmLangChainBridge: langchain })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const result = await run(structuredClone(arg))
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  const options = { modelId: 'test-model' }
  return {
    create,
    strands: browser ? new BrowserModel(page, options) : new WebLLMModel({ engine, ...options }),
    langchain: browser
      ? new BrowserChatWebLLM(page, options)
      : new ChatWebLLM({ engine, ...options }),
    close: () => Promise.all([strands.close(), langchain.close()]),
  }
}

it.each(clients)(
  '$framework (browser=$browser) does not execute native tool calls received after the response ended',
  async ({ framework, browser }) => {
    const client = setupModels(browser)
    const { create } = client
    create.mockResolvedValueOnce(
      chunks(
        { ...chunk({ content: 'Done' }, 'stop'), id: 'first-response' },
        {
          ...chunk({
            tool_calls: [
              {
                index: 0,
                id: 'late-call',
                function: { name: 'lookup', arguments: '{"code":"A7"}' },
              },
            ],
          }),
          id: 'first-response',
        },
      ),
    )
    const callback = vi.fn(() => 'Found')
    try {
      const invoke =
        framework === 'Strands'
          ? new Agent({
              model: client.strands,
              printer: false,
              tools: [
                new FunctionTool({ name: 'lookup', description: 'Look up a code.', callback }),
              ],
            }).invoke('Look up A7.')
          : createAgent({
              model: client.langchain,
              tools: [tool(callback, { name: 'lookup', schema: z.object({ code: z.string() }) })],
            }).invoke({ messages: [{ role: 'user', content: 'Look up A7.' }] })
      const failure = await invoke.catch((error: unknown) => error)

      expect(callback).not.toHaveBeenCalled()
      expect(failure).toHaveProperty('message', 'WebLLM returned content after its stop reason.')
      expect(create).toHaveBeenCalledTimes(1)
      expect((await client.langchain.invoke('Try again.')).text).toBe('Recovered')
    } finally {
      await client.close()
    }
  },
)

it.each(clients)(
  '$framework (browser=$browser) does not execute a valid first tool when a later native call is malformed',
  async ({ framework, browser }) => {
    const client = setupModels(browser)
    const { create } = client
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: 'first',
                function: { name: 'lookup', arguments: '{"code":"A7"}' },
              },
              { index: 1, id: 'second', function: { name: 'lookup', arguments: '{bad' } },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    const callback = vi.fn(() => 'Found')
    try {
      if (framework === 'Strands') {
        const agent = new Agent({
          model: client.strands,
          printer: false,
          tools: [new FunctionTool({ name: 'lookup', description: 'Look up a code.', callback })],
        })
        await expect(agent.invoke('Look up both codes.')).rejects.toBeInstanceOf(ModelError)
      } else {
        const agent = createAgent({
          model: client.langchain,
          tools: [tool(callback, { name: 'lookup', schema: z.object({ code: z.string() }) })],
        })
        await expect(
          agent.invoke({ messages: [{ role: 'user', content: 'Look up both codes.' }] }),
        ).rejects.toBeInstanceOf(SyntaxError)
      }
      expect(callback).not.toHaveBeenCalled()
      expect(create).toHaveBeenCalledTimes(1)
      expect((await client.langchain.invoke('Try again.')).text).toBe('Recovered')
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      await client.close()
    }
  },
)
