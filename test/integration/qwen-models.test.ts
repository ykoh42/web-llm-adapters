import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { Type } from '@earendil-works/pi-ai'
import { BrowserModel, WebLLMModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import { ModelType, postInitAndCheckFieldsChatCompletion, prebuiltAppConfig } from '@mlc-ai/web-llm'
import { chunk, chunks, collect, setupEngine } from '../../fixtures/web-llm.js'

const bases = new Set([
  'Qwen2.5-0.5B-Instruct',
  'Qwen2.5-1.5B-Instruct',
  'Qwen2.5-3B-Instruct',
  'Qwen2.5-7B-Instruct',
  'Qwen3-0.6B',
  'Qwen3-1.7B',
  'Qwen3-4B',
  'Qwen3-8B',
  'Qwen3.5-0.8B',
  'Qwen3.5-2B',
  'Qwen3.5-4B',
  'Qwen3.5-9B',
])
const modelIds = prebuiltAppConfig.model_list
  .filter(({ model_id }) => bases.has(model_id.replace(/-q[^-]+-MLC$/, '')))
  .map(({ model_id }) => model_id)
const schema = Type.Object({ code: Type.String() })
const tool = { name: 'lookup', description: 'Look up a code.', inputSchema: schema }
const definition = {
  type: 'function',
  function: { name: tool.name, description: tool.description, parameters: schema },
}

afterEach(() => vi.unstubAllGlobals())

for (const client of ['strands', 'strands-browser', 'langchain', 'langchain-browser', 'pi']) {
  it.each(modelIds)(`${client} uses the registered Qwen protocol for %s`, async (modelId) => {
    const thinking = !modelId.startsWith('Qwen2.5-')
    const call = modelId.startsWith('Qwen3.5-')
      ? '<tool_call><function=lookup><parameter=code>123</parameter></function></tool_call>'
      : '<tool_call>{"name":"lookup","arguments":{"code":"123"}}</tool_call>'
    const source = (thinking ? '<think>Plan.</think>' : '') + call
    const parts = source.match(/[\s\S]{1,3}/g)!
    const native = parts.map((content, index) => ({
      ...chunk({ content }, index === parts.length - 1 ? 'stop' : null),
      model: modelId,
    }))
    const { engine, create } = setupEngine()
    create.mockImplementation(async (request) => {
      postInitAndCheckFieldsChatCompletion(request, modelId, ModelType.LLM)
      return chunks(...native)
    })
    const bridge = client.endsWith('-browser')
      ? client === 'strands-browser'
        ? strandsBridge(engine)
        : langchainBridge(engine)
      : undefined
    vi.stubGlobal('window', { webllmBridge: bridge, webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        const result = await run(structuredClone(arg))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const onChunk = vi.fn()
    const options = {
      modelId,
      onChunk,
      request: { seed: 7, ...(thinking && { extra_body: { enable_thinking: true } }) },
    }
    async function run() {
      const input = 'Find code 123.'
      if (client === 'pi') {
        const provider = createWebLLMProvider({ engine, contextWindow: 4096, ...options })
        const response = await provider
          .stream(
            provider.getModels()[0]!,
            normalizeContext({
              messages: [{ role: 'user', content: input, timestamp: 0 }],
              tools: [{ name: tool.name, description: tool.description, parameters: schema }],
            }),
          )
          .result()
        if (response.stopReason === 'error') throw new Error(response.errorMessage)
        return {
          text: response.content
            .filter((part) => part.type === 'text')
            .map((part) => part.text)
            .join(''),
          thinking: response.content
            .filter((part) => part.type === 'thinking')
            .map((part) => part.thinking)
            .join(''),
          calls: response.content
            .filter((part) => part.type === 'toolCall')
            .map((part) => ({ name: part.name, args: part.arguments })),
        }
      }
      if (client.startsWith('strands')) {
        const model =
          client === 'strands'
            ? new WebLLMModel({ engine, ...options })
            : new BrowserModel(page, options)
        const blocks = await collect(
          model.streamAggregated([new Message({ role: 'user', content: [new TextBlock(input)] })], {
            toolSpecs: [tool],
          }),
        )
        return {
          text: blocks
            .filter((part) => part.type === 'textBlock')
            .map((part) => part.text)
            .join(''),
          thinking: blocks
            .filter((part) => part.type === 'reasoningBlock')
            .map((part) => part.text ?? '')
            .join(''),
          calls: blocks
            .filter((part) => part.type === 'toolUseBlock')
            .map((part) => ({ name: part.name, args: part.input })),
        }
      }
      const model =
        client === 'langchain'
          ? new ChatWebLLM({ engine, ...options })
          : new BrowserChatWebLLM(page, options)
      expect(model.profile).toMatchObject({ toolCalling: true, reasoningOutput: thinking })
      const response = await model.invoke(input, { tools: [definition] })
      return {
        text: response.text,
        thinking: response.contentBlocks
          .filter((part) => part.type === 'reasoning')
          .map((part) => part.reasoning)
          .join(''),
        calls: (response.tool_calls ?? []).map(({ name, args }) => ({ name, args })),
      }
    }
    try {
      expect(await run()).toEqual({
        text: '',
        thinking: thinking ? 'Plan.' : '',
        calls: [{ name: 'lookup', args: { code: '123' } }],
      })
      expect(create).toHaveBeenCalledOnce()
      expect(onChunk.mock.calls.map(([value]) => value)).toEqual(native)
      expect(create.mock.calls[0]?.[0].messages.at(-1)).toMatchObject({
        role: 'user',
        content: 'Find code 123.',
      })
    } finally {
      await bridge?.close()
    }
  })
}
