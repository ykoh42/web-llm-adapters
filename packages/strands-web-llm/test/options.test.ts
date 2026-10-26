import { afterEach, expect, it, vi } from 'vitest'
import {
  Agent,
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  type ChatCompletionRequestBase,
  type ResumableGenerationConfig,
  type ResponseFormat,
} from '@mlc-ai/web-llm'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge,
  withEngine,
  type BrowserPage,
  type WebLLMModelConfig,
} from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'
import { WebLLMModelBase } from '../src/model-base.js'

// Derived from WebLLM so a new native option requires an explicit coverage decision.
type NativeOptions = Required<
  Omit<ChatCompletionRequestBase, 'messages' | 'model' | 'stream' | 'tools' | 'n'>
>

const options = {
  max_tokens: 64,
  temperature: 0.2,
  top_p: 0.9,
  frequency_penalty: 0.1,
  presence_penalty: 0.2,
  repetition_penalty: 1.1,
  stop: ['done', 'finished'],
  logit_bias: { '7': -2 },
  logprobs: true,
  top_logprobs: 2,
  seed: 7,
  stream_options: { include_usage: true },
  tool_choice: 'none',
  response_format: { type: 'json_object', schema: '{"type":"object"}' },
  ignore_eos: false,
  extra_body: {
    enable_thinking: null,
    enable_latency_breakdown: true,
    resumable: {
      enabled: true,
      sessionId: 'native-session',
      durabilityMode: 'exact',
      strictPersistence: true,
      checkpointPrompt: true,
      checkpointIntervalTokens: 32,
      storeCheckpointLogits: true,
    } satisfies Required<ResumableGenerationConfig>,
  } satisfies Required<NonNullable<ChatCompletionRequestBase['extra_body']>>,
} satisfies NativeOptions

afterEach(() => vi.unstubAllGlobals())

function open(browser: boolean) {
  const native = setup(chunk({ content: '{}' }, 'stop'))
  const observed = vi.fn()
  const config = { modelId: 'test-model', request: structuredClone(options), onChunk: observed }
  const bridge = createBrowserBridge(native.engine)
  vi.stubGlobal('window', { webllmBridge: bridge })
  const page: BrowserPage = {
    evaluate: async (run, arg) => {
      const result = await run(arg)
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  const model = browser
    ? new BrowserModel(page, config)
    : new WebLLMModel({ engine: native.engine, ...config })
  return { ...native, model, bridge, page, observed }
}

it.each([false, true])(
  'keeps model identity current and config snapshots separate (browser: %s)',
  async (browser) => {
    const { model, create, bridge } = open(browser)
    const snapshot = model.getConfig()
    snapshot.modelId = 'changed-snapshot'
    snapshot.request!.logit_bias!['7'] = 99
    try {
      expect(model.modelId).toBe('test-model')
      expect(model.getConfig().request?.logit_bias).toEqual({ '7': -2 })
      model.updateConfig({ modelId: 'next-model' })
      expect(model.modelId).toBe('next-model')
      await new Agent({ model, printer: false }).invoke('Hello')
      expect(create.mock.calls[0]?.[0].model).toBe('next-model')
    } finally {
      await bridge.close()
    }
  },
)

it.each([false, true])(
  'uses an instance config reader for model identity and requests (browser: %s)',
  async (browser) => {
    const { model, create, bridge } = open(browser)
    const snapshot = model.getConfig()
    let id = 'custom-model'
    const read = vi
      .spyOn(model, 'getConfig')
      .mockImplementation(() => ({ ...snapshot, modelId: id }))
    try {
      expect(model.modelId).toBe('custom-model')
      id = 'next-custom-model'
      expect(model.modelId).toBe(id)
      await new Agent({ model, printer: false }).invoke('Hello')
      expect(create.mock.calls[0]?.[0].model).toBe(id)
    } finally {
      read.mockRestore()
      await bridge.close()
    }
  },
)

it.each([false, true])(
  'uses a subclass config reader for model identity and requests (browser: %s)',
  async (browser) => {
    const state = open(browser)
    let id = 'subclass-model'
    class Local extends WebLLMModel {
      override getConfig() {
        return { ...super.getConfig(), modelId: id }
      }
    }
    class Browser extends BrowserModel {
      override getConfig() {
        return { ...super.getConfig(), modelId: id }
      }
    }
    const config = state.model.getConfig()
    const model = browser
      ? new Browser(state.page, config)
      : new Local({ engine: state.engine, ...config })
    try {
      expect(model.modelId).toBe('subclass-model')
      id = 'next-subclass-model'
      expect(model.modelId).toBe(id)
      await new Agent({ model, printer: false }).invoke('Hello')
      expect(state.create.mock.calls[0]?.[0].model).toBe(id)
    } finally {
      await state.bridge.close()
    }
  },
)

it.each([false, true])(
  'preserves a custom config reader failure (browser: %s)',
  async (browser) => {
    const { model, create, bridge } = open(browser)
    const failure = new Error('Config is unavailable.')
    const read = vi.spyOn(model, 'getConfig').mockImplementation(() => {
      throw failure
    })
    try {
      expect(() => model.modelId).toThrow(failure)
      expect(create).not.toHaveBeenCalled()
    } finally {
      read.mockRestore()
      await bridge.close()
    }
  },
)

it('keeps prototype config readers observable', async () => {
  const { model, bridge } = open(false)
  const config = { ...model.getConfig(), modelId: 'patched-model' }
  const read = vi.spyOn(WebLLMModelBase.prototype, 'getConfig').mockReturnValue(config)
  try {
    expect(model.modelId).toBe('patched-model')
    expect(read).toHaveBeenCalledOnce()
  } finally {
    read.mockRestore()
    await bridge.close()
  }
  expect(model.modelId).toBe('test-model')
})

const gemmaId = 'gemma-4-E2B-it-q4f16_1-MLC'

it.each([false, true])(
  'forwards stream_options.include_usage=false (browser: %s)',
  async (browser) => {
    const { model, create, bridge } = open(browser)
    model.updateConfig({ request: { stream_options: { include_usage: false } } })
    try {
      await new Agent({ model, printer: false }).invoke('Hello')
      expect(create.mock.calls[0]?.[0].stream_options).toEqual({ include_usage: false })
    } finally {
      await bridge.close()
    }
  },
)

it.each(
  [false, true].flatMap((browser) =>
    [false, true].flatMap((frozen) => ['chat', 'raw'].map((path) => ({ browser, frozen, path }))),
  ),
)(
  'keeps shared Gemma config intact on $path (browser: $browser, frozen: $frozen)',
  async ({ browser, frozen, path }) => {
    const { model, engine, create, bridge } = open(browser)
    create.mockImplementation(async () => chunks(chunk({ content: 'Done' }, 'stop')))
    const complete = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield {
          id: 'raw',
          object: 'text_completion' as const,
          model: gemmaId,
          created: 0,
          choices: [{ index: 0, text: 'Done', finish_reason: 'stop' as const }],
        }
      })(),
    )
    const config: WebLLMModelConfig = {
      modelId: gemmaId,
      request: {
        seed: 7,
        extra_body: {
          enable_thinking: false,
          preserve_thinking: true,
          enable_latency_breakdown: true,
        },
      },
    }
    const original = structuredClone(config)
    if (frozen) {
      Object.freeze(config.request!.extra_body)
      Object.freeze(config.request)
      Object.freeze(config)
    }
    const read = vi.spyOn(model, 'getConfig').mockReturnValue(config)
    const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]
    const options =
      path === 'raw' ? { toolSpecs: [{ name: 'lookup', description: 'Read a value.' }] } : {}
    try {
      for (let turn = 0; turn < 2; turn++) {
        expect(await collect(model.stream(messages, options))).toContainEqual({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: 'Done' },
        })
        expect(config).toEqual(original)
      }
      const native = path === 'raw' ? complete : create
      expect(native).toHaveBeenCalledTimes(2)
      for (const [request] of native.mock.calls) {
        expect(request.model).toBe(gemmaId)
        expect(request.seed).toBe(7)
        expect(request.extra_body).toEqual({ enable_latency_breakdown: true })
      }
    } finally {
      read.mockRestore()
      await bridge.close()
    }
  },
)

const tokenConfigs = [
  { name: 'native to Gemma', before: { modelId: 'test-model' }, after: { modelId: gemmaId } },
  { name: 'Gemma to native', before: { modelId: gemmaId }, after: { modelId: 'test-model' } },
  {
    name: 'thinking enabled',
    before: { modelId: gemmaId },
    after: { modelId: gemmaId, request: { extra_body: { enable_thinking: true } } },
  },
  {
    name: 'preserved thoughts',
    before: { modelId: gemmaId },
    after: { modelId: gemmaId, request: { extra_body: { preserve_thinking: true } } },
  },
] satisfies { name: string; before: WebLLMModelConfig; after: WebLLMModelConfig }[]

it.each(tokenConfigs.flatMap((config) => [false, true].map((browser) => ({ ...config, browser }))))(
  'counts tokens with the current config: $name (browser: $browser)',
  async (config) => {
    const { model, engine, bridge } = open(config.browser)
    model.updateConfig({ ...config.before, request: {} })
    const read = vi.spyOn(model, 'getConfig').mockReturnValue(config.after)
    const history = [
      new Message({ role: 'user', content: [new TextBlock('Earlier question.')] }),
      new Message({
        role: 'assistant',
        content: [
          new ReasoningBlock({ text: 'Earlier plan. '.repeat(1000) }),
          new TextBlock('Done.'),
          new ToolUseBlock({ toolUseId: 'previous', name: 'lookup', input: {} }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'previous',
            status: 'success',
            content: [new TextBlock('42')],
          }),
        ],
      }),
      new Message({ role: 'user', content: [new TextBlock('Next question.')] }),
    ]
    const input = {
      systemPrompt: 'Answer the question.',
      toolSpecs: [{ name: 'lookup', description: 'Look up a value.' }],
    }
    const snapshot = JSON.stringify({ history, config: config.after })
    const reference = new WebLLMModel({ engine, ...config.after })
    try {
      expect(await model.countTokens(history, input)).toBe(
        await reference.countTokens(history, input),
      )
      expect(read).toHaveBeenCalledOnce()
      expect(JSON.stringify({ history, config: config.after })).toBe(snapshot)
    } finally {
      read.mockRestore()
      await bridge.close()
    }
  },
)

it.each([false, true])(
  'preserves config reader errors during token counting (browser: %s)',
  async (browser) => {
    const { model, bridge } = open(browser)
    const failure = new Error('Config is unavailable.')
    const read = vi.spyOn(model, 'getConfig').mockImplementation(() => {
      throw failure
    })
    try {
      await expect(model.countTokens([])).rejects.toBe(failure)
    } finally {
      read.mockRestore()
      await bridge.close()
    }
  },
)

it.each([false, true])(
  'preserves every native generation option through an Agent (browser: %s)',
  async (browser) => {
    const { model, create, bridge, observed } = open(browser)
    const first = chunk({ content: '{}' }, 'stop')
    first.choices[0]!.logprobs = {
      content: [{ token: '{}', logprob: -0.1, bytes: [123, 125], top_logprobs: [] }],
    }
    const usage = {
      prompt_tokens: 10,
      completion_tokens: 1,
      total_tokens: 11,
      extra: {
        e2e_latency_s: 0.1,
        prefill_tokens_per_s: 100,
        decode_tokens_per_s: 10,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.1,
        latencyBreakdown: {
          logitProcessorTime: [],
          logitBiasTime: [1],
          penaltyTime: [2],
          sampleTime: [3],
          totalTime: [6],
          grammarBitmaskTime: [0],
        },
      },
    }
    const last = { ...chunk(), choices: [], usage }
    create.mockResolvedValueOnce(chunks(first, last))
    const snapshot = model.getConfig()
    try {
      const result = await new Agent({
        model,
        printer: false,
        systemPrompt: 'Return JSON.',
      }).invoke('Hello')
      expect(result.lastMessage.content).toEqual([new TextBlock('{}')])
      expect(create.mock.calls[0]![0]).toEqual({
        ...options,
        model: 'test-model',
        stream: true,
        stream_options: { include_usage: true },
        n: 1,
        messages: [
          { role: 'system', content: 'Return JSON.' },
          { role: 'user', content: 'Hello' },
        ],
      })
      expect(observed.mock.calls.map(([value]) => value)).toEqual([first, last])
      expect(model.getConfig()).toEqual(snapshot)
    } finally {
      await bridge.close()
    }
  },
)

it.each([
  { type: 'text' },
  { type: 'json_object', schema: '{"type":"object"}' },
  { type: 'grammar', grammar: 'root ::= "yes" | "no"' },
  {
    type: 'structural_tag',
    structural_tag: { type: 'structural_tag', format: { type: 'const_string', value: 'yes' } },
  },
] satisfies ResponseFormat[])(
  'passes native output format %j through the browser boundary',
  async (response_format) => {
    const { model, create, bridge } = open(true)
    model.updateConfig({ request: { response_format } })
    try {
      await new Agent({ model, printer: false }).invoke('Return the requested format.')
      const request = create.mock.calls[0]![0]
      expect(request.response_format).toEqual(response_format)
      expect(() =>
        postInitAndCheckFieldsChatCompletion(structuredClone(request), 'test-model', ModelType.LLM),
      ).not.toThrow()
    } finally {
      await bridge.close()
    }
  },
)

const protocolModels = [
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
]

for (const browser of [false, true]) {
  it.each(protocolModels)(
    `keeps an active %s stream separate from updated config (browser: ${browser})`,
    async (modelId) => {
      const { model, engine, create, bridge } = open(browser)
      create.mockImplementation(async () => chunks(chunk({ content: '{}' }, 'stop')))
      const complete = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
        (async function* () {
          yield {
            id: 'raw',
            object: 'text_completion' as const,
            model: 'gemma-4-E2B-it-q4f16_1-MLC',
            created: 0,
            choices: [{ index: 0, text: '{}', finish_reason: 'stop' as const }],
          }
        })(),
      )
      const hold = pause()
      const active = withEngine(engine, () => hold.wait)
      const request = { seed: 7, extra_body: { enable_thinking: true } }
      model.updateConfig({ modelId, request, maxTokens: 64, temperature: 0, topP: 0.9 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Keep this input.')] })]
      const snapshots = vi.spyOn(model, 'getConfig')
      const pending = collect(model.stream(messages))
      try {
        await vi.waitFor(() => expect(snapshots).toHaveBeenCalledOnce())
        const nextModel = modelId.startsWith('gemma') ? protocolModels[0]! : protocolModels[2]!
        const nextRequest = { seed: 11, max_tokens: 1, extra_body: { enable_thinking: true } }
        model.updateConfig({ modelId: nextModel, request: nextRequest, maxTokens: 128 })
        request.seed = 99
        nextRequest.seed = 99
        expect(create).not.toHaveBeenCalled()
        expect(complete).not.toHaveBeenCalled()
        hold.resume()
        await active
        await pending
        await collect(model.stream(messages))
        const first = modelId.startsWith('gemma') ? complete : create
        const second = modelId.startsWith('gemma') ? create : complete
        expect(first).toHaveBeenCalledOnce()
        expect(second).toHaveBeenCalledOnce()
        expect(first.mock.calls[0]?.[0]).toMatchObject({
          model: modelId,
          max_tokens: 64,
          temperature: 0,
          top_p: 0.9,
          seed: 7,
        })
        expect(second.mock.calls[0]?.[0]).toMatchObject({
          model: nextModel,
          max_tokens: 128,
          temperature: 0,
          top_p: 0.9,
          seed: 11,
        })
        expect(model.getConfig().request?.seed).toBe(11)
      } finally {
        hold.resume()
        await active
        await pending.catch(() => {})
        await bridge.close()
      }
    },
  )
}
