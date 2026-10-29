import { afterEach, expect, it, vi } from 'vitest'
import { Agent, BeforeModelCallEvent, Message, TextBlock } from '@strands-agents/sdk'
import { MLCEngine, type Completion } from '@mlc-ai/web-llm'
import {
  BrowserModel,
  createBrowserBridge,
  WebLLMModel,
  type BrowserPage,
} from '../../src/index.js'
import { collect, setup } from '../../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

it.each([false, true])(
  'counts only new messages after a Gemma response (browser: %s)',
  async (browser) => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    const engine = new MLCEngine({ logLevel: 'SILENT' })
    const answer: Completion = {
      id: 'test',
      object: 'text_completion',
      model: modelId,
      created: 0,
      choices: [{ index: 0, text: 'Done', finish_reason: 'stop' }],
    }
    const usage: Completion = {
      ...answer,
      choices: [],
      usage: {
        prompt_tokens: 100,
        completion_tokens: 10,
        total_tokens: 110,
        extra: {
          e2e_latency_s: 0,
          prefill_tokens_per_s: 0,
          decode_tokens_per_s: 0,
          time_to_first_token_s: 0,
          time_per_output_token_s: 0,
        },
      },
    }
    vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield answer
        yield usage
      })(),
    )
    const bridge = createBrowserBridge(engine)
    vi.stubGlobal('window', { webllmBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, input) {
        const result = await run(structuredClone(input))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const config = {
      modelId,
      contextWindowLimit: 8192,
      request: { extra_body: { enable_thinking: true } },
    }
    const model = browser ? new BrowserModel(page, config) : new WebLLMModel({ engine, ...config })
    const agent = new Agent({ model, systemPrompt: '', contextManager: 'auto', printer: false })
    try {
      await collect(agent.stream('First'))
      const events = await collect(agent.stream('Next'))
      const before = events.find((event) => event instanceof BeforeModelCallEvent)
      const messages = [new Message({ role: 'user', content: [new TextBlock('Next')] })]
      const extra = await setup().model.countTokens(messages)
      expect(before?.projectedInputTokens).toBe(110 + extra)
      expect(await model.countTokens(messages, {})).toBeGreaterThan(extra)
    } finally {
      await bridge.close()
    }
  },
)
