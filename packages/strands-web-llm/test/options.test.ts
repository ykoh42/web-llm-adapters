import { afterEach, expect, it, vi } from 'vitest'
import { Agent, TextBlock } from '@strands-agents/sdk'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  type ChatCompletionRequestBase,
  type ResponseFormat,
} from '@mlc-ai/web-llm'
import { BrowserModel, WebLLMModel, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, chunks, setup } from './helpers.js'

// Derived from WebLLM so a new native option requires an explicit coverage decision.
type NativeOptions = Required<
  Omit<
    ChatCompletionRequestBase,
    'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'n'
  >
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
    },
  },
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
  return { ...native, model, bridge, observed }
}

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
