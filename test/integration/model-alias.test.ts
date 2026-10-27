import { afterEach, expect, it, vi } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import { InMemoryCache } from '@langchain/core/caches'
import { Message, TextBlock, ToolResultBlock } from '@strands-agents/sdk'
import { BrowserModel, WebLLMModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import { chunk, chunks, collect, setupEngine } from '../../fixtures/web-llm.js'

const sources = [
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
]
const clients = ['strands', 'strands-browser', 'langchain', 'langchain-browser', 'pi'] as const
const schema = Type.Object({ code: Type.String() }, { additionalProperties: false })
const tool = { name: 'lookup', description: 'Look up a code.', inputSchema: schema }
const definition = {
  type: 'function',
  function: { name: tool.name, description: tool.description, parameters: schema },
}
const user = 'Look up A7.'
const alias = 'my-model'

afterEach(() => vi.unstubAllGlobals())

async function read(stream: ReturnType<WebLLMModel['streamAggregated']>) {
  for (;;) {
    const next = await stream.next()
    if (next.done) return next.value.message.content
  }
}

function replies(source: string) {
  const xml = source.startsWith('Qwen3.5-')
  const gemma = source.startsWith('gemma-')
  const thinking = !source.startsWith('Qwen2.5-')
  const call = gemma
    ? '<|channel>thought\nPlan<channel|><|tool_call>call:lookup{code:<|"|>A7<|"|>}<tool_call|>'
    : (thinking ? (xml ? 'Plan</think>' : '<think>Plan</think>') : '') +
      (xml
        ? '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
        : '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>')
  const answer = gemma
    ? 'Done<channel|>The answer is 42.'
    : (thinking ? (xml ? 'Done</think>' : '<think>Done</think>') : '') + 'The answer is 42.'
  return [call, answer]
}

async function run(
  client: (typeof clients)[number],
  source: string,
  custom: boolean,
  constrained = false,
) {
  const modelId = custom ? alias : source
  const [call, answer] = replies(source)
  const { engine, create } = setupEngine()
  const raw = vi.spyOn(engine.completions, 'create')
  const outputs = [call, answer]
  create.mockImplementation(async () => chunks(chunk({ content: outputs.shift()! }, 'stop')))
  raw.mockImplementation(async () =>
    (async function* () {
      yield {
        id: 'raw',
        object: 'text_completion' as const,
        created: 0,
        model: modelId,
        choices: [
          { index: 0, text: outputs.shift()!, finish_reason: 'stop' as const, logprobs: null },
        ],
      }
    })(),
  )
  const bridge = client.endsWith('-browser')
    ? client === 'strands-browser'
      ? strandsBridge(engine)
      : langchainBridge(engine)
    : undefined
  vi.stubGlobal('window', { webllmBridge: bridge, webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(evaluate, arg) {
      const value = await evaluate(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  const options = {
    modelId,
    ...(custom && { sourceModelId: source }),
    thinkingPrefilled: source.startsWith('Qwen3.5-'),
    request: {
      extra_body: { enable_thinking: !source.startsWith('Qwen2.5-') },
      ...(constrained && { constrainedToolCalls: true }),
    },
  }
  let result: { thinking: string; name: string; args: unknown; answer: string }
  try {
    if (client.startsWith('strands')) {
      const model =
        client === 'strands'
          ? new WebLLMModel({ engine, ...options })
          : new BrowserModel(page, options)
      const messages = [new Message({ role: 'user', content: [new TextBlock(user)] })]
      const first = await read(model.streamAggregated(messages, { toolSpecs: [tool] }))
      const selected = first.find((block) => block.type === 'toolUseBlock')!
      if (selected.type !== 'toolUseBlock') throw new Error('Missing tool call.')
      const final = await read(
        model.streamAggregated(
          [
            ...messages,
            new Message({ role: 'assistant', content: first }),
            new Message({
              role: 'user',
              content: [
                new ToolResultBlock({
                  toolUseId: selected.toolUseId,
                  status: 'success',
                  content: [new TextBlock('42')],
                }),
              ],
            }),
          ],
          { toolSpecs: [tool] },
        ),
      )
      result = {
        thinking: first
          .filter((block) => block.type === 'reasoningBlock')
          .map((block) => block.text ?? '')
          .join(''),
        name: selected.name,
        args: selected.input,
        answer: final
          .filter((block) => block.type === 'textBlock')
          .map((block) => block.text)
          .join(''),
      }
    } else if (client.startsWith('langchain')) {
      const model =
        client === 'langchain'
          ? new ChatWebLLM({ engine, ...options })
          : new BrowserChatWebLLM(page, options)
      expect(model.profile.toolCalling).toBe(true)
      const first = await model.invoke(user, { tools: [definition] })
      const selected = first.tool_calls![0]!
      const final = await model.invoke(
        [
          new HumanMessage(user),
          first,
          new ToolMessage({ tool_call_id: selected.id!, content: '42' }),
        ],
        { tools: [definition] },
      )
      result = {
        thinking: first.contentBlocks
          .filter((block) => block.type === 'reasoning')
          .map((block) => block.reasoning)
          .join(''),
        name: selected.name,
        args: selected.args,
        answer: final.text,
      }
    } else {
      const provider = createWebLLMProvider({ engine, contextWindow: 8192, ...options })
      const models = createModels()
      models.setProvider(provider)
      const model = provider.getModels()[0]!
      const messages = [{ role: 'user' as const, content: user, timestamp: 0 }]
      const tools = [{ name: tool.name, description: tool.description, parameters: schema }]
      const first = await models.complete(model, { messages, tools })
      const selected = first.content.find((part) => part.type === 'toolCall')!
      if (selected.type !== 'toolCall') throw new Error('Missing tool call.')
      const final = await models.complete(model, {
        messages: [
          ...messages,
          first,
          {
            role: 'toolResult',
            toolCallId: selected.id,
            toolName: selected.name,
            content: [{ type: 'text', text: '42' }],
            isError: false,
            timestamp: 0,
          },
        ],
        tools,
      })
      result = {
        thinking: first.content
          .filter((part) => part.type === 'thinking')
          .map((part) => part.thinking)
          .join(''),
        name: selected.name,
        args: selected.arguments,
        answer: final.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join(''),
      }
    }
    const requests = source.startsWith('gemma-') ? raw.mock.calls : create.mock.calls
    expect(requests).toHaveLength(2)
    return { result, requests: requests.map(([request]) => request) }
  } finally {
    await bridge?.close()
  }
}

for (const client of clients) {
  it.each(sources)(
    `${client} preserves %s tool and thinking output with a custom ID`,
    async (source) => {
      const original = await run(client, source, false)
      const renamed = await run(client, source, true)
      expect(renamed.result).toEqual(original.result)
      expect(renamed.result).toEqual({
        thinking: source.startsWith('Qwen2.5-') ? '' : 'Plan',
        name: 'lookup',
        args: { code: 'A7' },
        answer: 'The answer is 42.',
      })
      expect(renamed.requests).toEqual(
        original.requests.map((request) => ({ ...request, model: alias })),
      )
    },
  )

  it(`${client} keeps verified Qwen constraints for an alias`, async () => {
    const source = 'Qwen3.5-4B-q4f16_1-MLC'
    const original = await run(client, source, false, true)
    const renamed = await run(client, source, true, true)
    expect(renamed.result).toEqual(original.result)
    expect(renamed.requests).toEqual(
      original.requests.map((request) => ({ ...request, model: alias })),
    )
    expect(renamed.requests[0]).toHaveProperty('response_format.type', 'structural_tag')
  })
}

it('rejects an unknown source before using an engine', () => {
  const { engine } = setupEngine()
  const options = { engine, modelId: alias, sourceModelId: 'unknown-model' }
  expect(() => new WebLLMModel(options)).toThrow('sourceModelId')
  expect(() => new ChatWebLLM(options)).toThrow('sourceModelId')
  expect(() => createWebLLMProvider({ ...options, contextWindow: 8192 })).toThrow('sourceModelId')
})

it('does not enable constraints for an unverified source model', async () => {
  const { engine } = setupEngine()
  const options = {
    engine,
    modelId: alias,
    sourceModelId: 'Qwen3.5-4B-q4f32_1-MLC',
    request: { constrainedToolCalls: true },
  }
  const strands = new WebLLMModel(options)
  await expect(
    collect(
      strands.stream([new Message({ role: 'user', content: [new TextBlock(user)] })], {
        toolSpecs: [tool],
      }),
    ),
  ).rejects.toThrow('not verified')
  await expect(new ChatWebLLM(options).invoke(user, { tools: [definition] })).rejects.toThrow(
    'not verified',
  )
  const provider = createWebLLMProvider({ ...options, contextWindow: 8192 })
  const models = createModels()
  models.setProvider(provider)
  const result = await models.complete(provider.getModels()[0]!, {
    messages: [{ role: 'user', content: user, timestamp: 0 }],
    tools: [{ name: tool.name, description: tool.description, parameters: schema }],
  })
  expect(result.stopReason).toBe('error')
  expect(result.errorMessage).toContain('not verified')
})

it('leaves an unknown model on WebLLM native chat when no source is given', async () => {
  const { engine, create } = setupEngine(chunk({ content: 'Native answer' }, 'stop'))
  const response = await new ChatWebLLM({ engine, modelId: alias }).invoke('Hi')
  expect(response.text).toBe('Native answer')
  expect(create.mock.calls[0]?.[0]).toMatchObject({
    model: alias,
    messages: [{ role: 'user', content: 'Hi' }],
  })
})

it('requires a Strands alias to be set or cleared when the loaded model changes', () => {
  const { engine } = setupEngine()
  const model = new WebLLMModel({
    engine,
    modelId: alias,
    sourceModelId: 'Qwen3-4B-q4f16_1-MLC',
  })
  expect(() => model.updateConfig({ modelId: 'another-model' })).toThrow('sourceModelId')
  expect(model.getConfig().modelId).toBe(alias)
  model.updateConfig({ modelId: 'another-model', sourceModelId: undefined })
  expect(model.getConfig().sourceModelId).toBeUndefined()
  model.updateConfig({ modelId: alias, sourceModelId: 'Qwen3.5-4B-q4f16_1-MLC' })
  expect(model.getConfig().sourceModelId).toBe('Qwen3.5-4B-q4f16_1-MLC')
})

it('keeps different source models separate in the LangChain cache', async () => {
  const { engine, create } = setupEngine()
  create
    .mockResolvedValueOnce(chunks(chunk({ content: 'First' }, 'stop')))
    .mockResolvedValueOnce(chunks(chunk({ content: 'Second' }, 'stop')))
  const cache = new InMemoryCache()
  const first = new ChatWebLLM({
    engine,
    cache,
    modelId: alias,
    sourceModelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  })
  const second = new ChatWebLLM({
    engine,
    cache,
    modelId: alias,
    sourceModelId: 'Qwen3.5-4B-q4f16_1-MLC',
  })
  expect((await first.invoke('Hi')).text).toBe('First')
  expect((await second.invoke('Hi')).text).toBe('Second')
  expect((await first.invoke('Hi')).text).toBe('First')
  expect(create).toHaveBeenCalledTimes(2)
})
