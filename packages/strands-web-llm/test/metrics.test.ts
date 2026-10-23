import { afterEach, expect, it, vi } from 'vitest'
import { Agent, Message, TextBlock } from '@strands-agents/sdk'
import { BrowserModel, WebLLMModel, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, chunks, collect, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

function response(latency: number) {
  return {
    ...chunk({ content: 'Hello' }, 'stop'),
    usage: {
      prompt_tokens: 4,
      completion_tokens: 1,
      total_tokens: 5,
      extra: {
        e2e_latency_s: latency,
        prefill_tokens_per_s: 20,
        decode_tokens_per_s: 10,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.1,
      },
    },
  }
}

for (const browser of [false, true]) {
  it.each([0, 0.1255, 1.25])(
    `reports native latency %s in SDK milliseconds (browser: ${browser})`,
    async (latency) => {
      const native = response(latency)
      const { engine, create } = setup()
      create.mockImplementation(async () => chunks(native))
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const model = browser
        ? new BrowserModel(page, { modelId: 'test-model' })
        : new WebLLMModel({ engine, modelId: 'test-model' })
      try {
        const events = await collect(model.stream(messages))
        const metadata = events.find((event) => event.type === 'modelMetadataEvent')
        expect(metadata?.metrics).toEqual({ latencyMs: latency * 1000 })
        expect(metadata?.trace).toEqual(native.usage.extra)
        const agent = new Agent({ model, printer: false })
        const answer = await agent.invoke('Hello')
        expect(answer.lastMessage.content).toEqual([new TextBlock('Hello')])
        expect(answer.metrics?.accumulatedMetrics.latencyMs).toBe(latency * 1000)
        expect(answer.invocationContext?.usage).toEqual({
          turnsTaken: 1,
          inputTokens: 4,
          outputTokens: 1,
          totalTokens: 5,
        })
        const followup = await agent.invoke('Hello again', {
          invocationContext: answer.invocationContext!,
        })
        expect(followup.invocationContext).toBe(answer.invocationContext)
        expect(followup.invocationContext?.usage).toEqual({
          turnsTaken: 2,
          inputTokens: 8,
          outputTokens: 2,
          totalTokens: 10,
        })
      } finally {
        await bridge.close()
      }
    },
  )
}

it.each([-1, NaN, Infinity, Number.MAX_VALUE])(
  'does not report invalid native latency %s',
  async (latency) => {
    const { model } = setup(response(latency))
    const events = await collect(model.stream(messages))
    const metadata = events.find((event) => event.type === 'modelMetadataEvent')
    expect(metadata).not.toHaveProperty('metrics')
    expect(metadata?.usage).toEqual({ inputTokens: 4, outputTokens: 1, totalTokens: 5 })
  },
)

it('keeps usage without inventing latency when native timing is absent', async () => {
  const native = response(0)
  Reflect.deleteProperty(native.usage, 'extra')
  const { model } = setup(native)
  const events = await collect(model.stream(messages))
  const metadata = events.find((event) => event.type === 'modelMetadataEvent')
  expect(metadata).not.toHaveProperty('metrics')
  expect(metadata?.usage).toEqual({ inputTokens: 4, outputTokens: 1, totalTokens: 5 })
})
