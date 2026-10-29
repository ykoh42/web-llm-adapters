import { expect, it, vi } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { MLCEngine, type Completion } from '@mlc-ai/web-llm'
import { Message, TextBlock } from '@strands-agents/sdk'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge as createStrandsBridge,
} from '../../packages/strands-web-llm/src/index.js'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as createLangChainBridge,
  type BrowserPage,
} from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { collect } from '../../fixtures/web-llm.js'

it('preserves negative zero in Gemma tool arguments across the frameworks', async () => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  const schema = {
    type: 'object' as const,
    properties: { value: { type: 'number' as const } },
    required: ['value'],
  }
  const tool = { name: 'lookup', description: 'Look up a value.', inputSchema: schema }
  const response: Completion = {
    id: 'test',
    model: modelId,
    object: 'text_completion',
    created: 0,
    choices: [
      {
        index: 0,
        text: '<|tool_call>call:lookup{value:-0.0}<tool_call|>',
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
  }
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
    (async function* () {
      yield response
    })(),
  )

  const strands = new WebLLMModel({ engine, modelId })
  const events = await collect(strands.stream([], { toolSpecs: [tool] }))
  const input = events
    .filter((event) => event.type === 'modelContentBlockDeltaEvent')
    .map((event) => (event.delta.type === 'toolUseInputDelta' ? event.delta.input : ''))
    .join('')
  expect(Object.is(JSON.parse(input).value, -0)).toBe(true)

  const langchain = new ChatWebLLM({ engine, modelId })
  const answer = await langchain
    .bindTools([{ type: 'function', function: { name: tool.name, parameters: schema } }])
    .invoke('Look up the value.')
  expect(Object.is(answer.tool_calls?.[0]?.args.value, -0)).toBe(true)

  const strandsBridge = createStrandsBridge(engine)
  const langchainBridge = createLangChainBridge(engine)
  vi.stubGlobal('window', {
    webllmBridge: strandsBridge,
    webllmLangChainBridge: langchainBridge,
  })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  try {
    const browserStrands = new BrowserModel(page, { modelId })
    const events = await collect(
      browserStrands.stream(
        [new Message({ role: 'user', content: [new TextBlock('Look up the value.')] })],
        {
          toolSpecs: [tool],
        },
      ),
    )
    const input = events
      .filter((event) => event.type === 'modelContentBlockDeltaEvent')
      .map((event) => (event.delta.type === 'toolUseInputDelta' ? event.delta.input : ''))
      .join('')
    expect(Object.is(JSON.parse(input).value, -0)).toBe(true)

    const browserLangChain = new BrowserChatWebLLM(page, { modelId })
    const answer = await browserLangChain
      .bindTools([{ type: 'function', function: { name: tool.name, parameters: schema } }])
      .invoke('Look up the value.')
    expect(Object.is(answer.tool_calls?.[0]?.args.value, -0)).toBe(true)
  } finally {
    await strandsBridge.close()
    await langchainBridge.close()
    vi.unstubAllGlobals()
  }

  const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
  const models = createModels()
  models.setProvider(provider)
  const answerPi = await models.complete(provider.getModels()[0]!, {
    messages: [{ role: 'user', content: 'Look up the value.', timestamp: 0 }],
    tools: [
      {
        name: tool.name,
        description: tool.description,
        parameters: Type.Object({ value: Type.Number() }),
      },
    ],
  })
  const call = answerPi.content.find((part) => part.type === 'toolCall')
  expect(call?.type).toBe('toolCall')
  if (call?.type !== 'toolCall') throw new Error('Missing tool call.')
  expect(Object.is(call.arguments.value, -0)).toBe(true)
})
