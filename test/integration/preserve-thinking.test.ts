import { afterEach, expect, it, vi } from 'vitest'
import {
  Message,
  TextBlock,
  ReasoningBlock,
  ToolUseBlock,
  ToolResultBlock,
  Agent,
  ModelError,
} from '@strands-agents/sdk'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { Type, type AssistantMessage } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge as strandsBridge,
  type BrowserPage,
} from '../../packages/strands-web-llm/src/index.js'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
} from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import type { WebLLMRequest } from '../../shared/web-llm/request.js'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

const gemma = 'gemma-4-E2B-it-q4f16_1-MLC'
const clients = ['strands', 'strands browser', 'langchain', 'langchain browser', 'pi'] as const
type Client = (typeof clients)[number]
const tool = {
  name: 'lookup',
  description: 'Look up a code.',
  inputSchema: { type: 'object' as const, properties: { code: { type: 'string' as const } } },
}
const nativeTool = {
  type: 'function',
  function: { name: tool.name, description: tool.description, parameters: tool.inputSchema },
}

afterEach(() => vi.unstubAllGlobals())

function assistant(model: string, content: AssistantMessage['content']): AssistantMessage {
  return {
    role: 'assistant',
    api: 'webllm',
    provider: 'webllm',
    model,
    content,
    stopReason: content.some((part) => part.type === 'toolCall') ? 'toolUse' : 'stop',
    timestamp: 0,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  }
}

function histories(modelId: string) {
  const strands: Message[] = []
  const langchain: (HumanMessage | AIMessage | ToolMessage)[] = []
  const pi = []
  for (const [index, thought] of ['Old tool plan', 'Current tool plan'].entries()) {
    const id = `call-${index}`
    const question = index ? 'Next question' : 'First question'
    strands.push(
      new Message({ role: 'user', content: [new TextBlock(question)] }),
      new Message({
        role: 'assistant',
        content: [
          new ReasoningBlock({ text: thought }),
          new ToolUseBlock({ toolUseId: id, name: tool.name, input: { code: 'A7' } }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: id,
            content: [new TextBlock('42')],
            status: 'success',
          }),
        ],
      }),
    )
    langchain.push(
      new HumanMessage(question),
      new AIMessage({
        content: [{ type: 'reasoning', reasoning: thought }],
        tool_calls: [{ id, name: tool.name, args: { code: 'A7' } }],
      }),
      new ToolMessage({ content: '42', tool_call_id: id }),
    )
    pi.push(
      { role: 'user' as const, content: question, timestamp: 0 },
      assistant(modelId, [
        { type: 'thinking', thinking: thought },
        { type: 'toolCall', id, name: tool.name, arguments: { code: 'A7' } },
      ]),
      {
        role: 'toolResult' as const,
        toolCallId: id,
        toolName: tool.name,
        content: [{ type: 'text' as const, text: '42' }],
        isError: false,
        timestamp: 0,
      },
    )
    if (!index) {
      strands.push(
        new Message({
          role: 'assistant',
          content: [new ReasoningBlock({ text: 'Old final plan' }), new TextBlock('42')],
        }),
      )
      langchain.push(
        new AIMessage({
          content: [
            { type: 'reasoning', reasoning: 'Old final plan' },
            { type: 'text', text: '42' },
          ],
        }),
      )
      pi.push(
        assistant(modelId, [
          { type: 'thinking', thinking: 'Old final plan' },
          { type: 'text', text: '42' },
        ]),
      )
    }
  }
  return { strands, langchain, pi }
}

function open(
  client: Client,
  request: WebLLMRequest,
  modelId = gemma,
  override?: WebLLMRequest,
  bound = false,
  samplingParams?: Record<string, unknown>,
) {
  const { engine, create } = setupEngine(chunk({ content: 'Done' }, 'stop'))
  const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
    (async function* () {
      yield {
        id: 'test',
        model: modelId,
        object: 'text_completion' as const,
        created: 0,
        choices: [{ index: 0, text: '<channel|>Done', finish_reason: 'stop', logprobs: null }],
      }
    })(),
  )
  const history = histories(modelId)
  const config = { modelId, request }
  const bridge =
    client === 'strands browser'
      ? strandsBridge(engine)
      : client === 'langchain browser'
        ? langchainBridge(engine)
        : undefined
  if (bridge)
    vi.stubGlobal('window', {
      webllmBridge: bridge,
      webllmLangChainBridge: bridge,
    })
  const page: BrowserPage = {
    evaluate: async (run, arg) => {
      const result = await run(arg)
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  async function run() {
    try {
      if (client === 'pi') {
        const provider = createWebLLMProvider({ engine, ...config, contextWindow: 32768 })
        const models = createModels()
        models.setProvider(provider)
        const model = provider.getModels()[0]!
        if (samplingParams) model.samplingParams = samplingParams
        const result = await models.complete(
          model,
          {
            tools: [
              {
                name: tool.name,
                description: tool.description,
                parameters: Type.Unsafe<{ code: string }>(tool.inputSchema),
              },
            ],
            messages: history.pi,
          },
          override && { request: override },
        )
        if (result.stopReason === 'error') throw new Error(result.errorMessage)
      } else if (client.startsWith('langchain')) {
        const model =
          client === 'langchain browser'
            ? new BrowserChatWebLLM(page, config)
            : new ChatWebLLM({ engine, ...config })
        const call = override && { request: override }
        await model
          .bindTools([nativeTool], bound ? call : undefined)
          .invoke(history.langchain, bound ? undefined : call)
      } else {
        const model =
          client === 'strands browser'
            ? new BrowserModel(page, config)
            : new WebLLMModel({ engine, ...config })
        if (override) model.updateConfig({ request: override })
        await collect(model.stream(history.strands, { toolSpecs: [tool] }))
      }
    } finally {
      await bridge?.close()
    }
  }
  return { run, raw, create, history }
}

for (const client of clients) {
  it.each([undefined, null, false, true])(
    `${client} preserves only historical Gemma tool-call reasoning with %s`,
    async (preserve_thinking) => {
      const request = {
        extra_body: {
          enable_thinking: true,
          ...(preserve_thinking !== undefined && { preserve_thinking }),
        },
      }
      const { run, raw, history } = open(client, request)
      const snapshot = JSON.stringify({ request, history })
      await run()
      const sent = raw.mock.calls[0]![0]
      expect(sent.prompt.includes('Old tool plan')).toBe(preserve_thinking === true)
      expect(sent.prompt).not.toContain('Old final plan')
      expect(sent.prompt).toContain('Current tool plan')
      expect(sent).not.toHaveProperty('extra_body.preserve_thinking')
      expect(JSON.stringify({ request, history })).toBe(snapshot)
      if (preserve_thinking !== true) {
        const baseline = open(client, { extra_body: { enable_thinking: true } })
        await baseline.run()
        expect(sent).toEqual(baseline.raw.mock.calls[0]![0])
      }
    },
  )

  it.each([undefined, false])(
    `${client} preserves history without changing generation thinking (%s)`,
    async (enable_thinking) => {
      const { run, raw } = open(client, {
        extra_body: {
          preserve_thinking: true,
          ...(enable_thinking !== undefined && { enable_thinking }),
        },
      })
      await run()
      const sent = raw.mock.calls[0]![0]
      expect(sent.prompt).toContain('Old tool plan')
      expect(sent.prompt).toContain('Current tool plan')
      expect(sent.prompt).not.toContain('Old final plan')
      expect(sent.prompt).not.toContain('<|think|>')
      expect(sent.prompt.endsWith('<|channel>thought\n')).toBe(false)
      expect(sent).not.toHaveProperty('extra_body.preserve_thinking')
    },
  )

  it.each([false, true])(`${client} respects a %s per-call or updated option`, async (preserve) => {
    const request = { extra_body: { enable_thinking: true, preserve_thinking: !preserve } }
    const override = { extra_body: { enable_thinking: true, preserve_thinking: preserve } }
    const { run, raw } = open(client, request, gemma, override)
    await run()
    expect(raw.mock.calls[0]![0].prompt.includes('Old tool plan')).toBe(preserve)
  })

  it.each([false, true])(`${client} rejects %s preservation on Qwen`, async (preserve_thinking) => {
    const { run, create, raw } = open(
      client,
      { extra_body: { preserve_thinking } },
      'Qwen3.5-4B-q4f16_1-MLC',
    )
    await expect(run()).rejects.toThrow('does not support preserve_thinking')
    expect(create).not.toHaveBeenCalled()
    expect(raw).not.toHaveBeenCalled()
  })

  it(`${client} consumes null preservation on the native path`, async () => {
    const { run, create } = open(client, { extra_body: { preserve_thinking: null } }, 'test-model')
    await run()
    expect(create.mock.calls[0]![0]).not.toHaveProperty('extra_body.preserve_thinking')
  })

  it.each(['true', 1, {}])(`${client} rejects invalid preservation %j`, async (value) => {
    const request: WebLLMRequest = {}
    Object.assign(request, { extra_body: { preserve_thinking: value } })
    const { run, create, raw } = open(client, request)
    await expect(run()).rejects.toThrow('preserve_thinking must be a boolean or null')
    expect(create).not.toHaveBeenCalled()
    expect(raw).not.toHaveBeenCalled()
  })
}

it.each(['langchain', 'langchain browser'] as const)(
  '%s forwards bound preservation options',
  async (client) => {
    const { run, raw } = open(
      client,
      { extra_body: { enable_thinking: true, preserve_thinking: false } },
      gemma,
      { extra_body: { enable_thinking: true, preserve_thinking: true } },
      true,
    )
    await run()
    expect(raw.mock.calls[0]![0].prompt).toContain('Old tool plan')
    expect(raw.mock.calls[0]![0].prompt).not.toContain('Old final plan')
  },
)

it.each([false, true])(
  'Strands reports invalid preservation through the SDK error contract (browser: %s)',
  async (browser) => {
    const { engine, create } = setupEngine()
    const raw = vi.spyOn(engine.completions, 'create')
    const request: WebLLMRequest = {}
    Object.assign(request, { extra_body: { preserve_thinking: 'true' } })
    const config = { modelId: gemma, request }
    const bridge = strandsBridge(engine)
    vi.stubGlobal('window', { webllmBridge: bridge })
    const page: BrowserPage = { evaluate: async (run, arg) => run(arg) }
    const model = browser ? new BrowserModel(page, config) : new WebLLMModel({ engine, ...config })
    try {
      const result = new Agent({ model, printer: false }).invoke('Hello')
      await expect(result).rejects.toBeInstanceOf(ModelError)
      await expect(result).rejects.toHaveProperty('cause', expect.any(Error))
      await expect(result).rejects.toThrow('preserve_thinking must be a boolean or null')
      expect(create).not.toHaveBeenCalled()
      expect(raw).not.toHaveBeenCalled()
    } finally {
      await bridge.close()
    }
  },
)

it.each([false, true])(
  'Strands counts retained Gemma tool-call reasoning (browser: %s)',
  async (browser) => {
    const { engine } = setupEngine()
    const config = { modelId: gemma, request: { extra_body: { preserve_thinking: true } } }
    const page: BrowserPage = {
      evaluate: () => Promise.reject(new Error('No inference expected.')),
    }
    const model = browser ? new BrowserModel(page, config) : new WebLLMModel({ engine, ...config })
    const history = histories(gemma).strands
    const snapshot = JSON.stringify(history)
    const retained = history.map((message, index) =>
      index === 3
        ? new Message({
            role: message.role,
            content: message.content.filter((block) => block.type !== 'reasoningBlock'),
          })
        : message,
    )
    expect(await model.countTokens(history)).toBe(await model.countTokens(retained))
    model.updateConfig({ request: { extra_body: { preserve_thinking: false } } })
    expect(await model.countTokens(history)).toBeLessThan(
      await new WebLLMModel({ engine, ...config }).countTokens(history),
    )
    expect(JSON.stringify(history)).toBe(snapshot)
  },
)

it('Pi retains Gemma tool reasoning when native sampling options are set', async () => {
  const { run, raw } = open(
    'pi',
    { extra_body: { enable_thinking: true, preserve_thinking: true } },
    gemma,
    undefined,
    false,
    { extra_body: { enable_latency_breakdown: true } },
  )

  await run()

  const sent = raw.mock.calls[0]![0]
  expect(sent.prompt).toContain('Old tool plan')
  expect(sent.prompt).toContain('Current tool plan')
  expect(sent).not.toHaveProperty('extra_body.preserve_thinking')
  expect(sent.extra_body).toEqual({ enable_latency_breakdown: true })
})
