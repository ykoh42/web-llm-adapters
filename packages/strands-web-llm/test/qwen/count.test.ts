import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  Agent,
  BeforeModelCallEvent,
  FunctionTool,
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import { BrowserModel, createBrowserBridge, type BrowserPage } from '../../src/index.js'
import { chunk, chunks, collect, setup } from '../../fixtures/engine.js'

const models = ['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']

afterEach(() => vi.unstubAllGlobals())

for (const browser of [false, true]) {
  describe(`Qwen token estimates (browser: ${browser})`, () => {
    it.each([undefined, '', 'Be concise.'])(
      'counts the Qwen2.5 system prompt actually sent: %j',
      async (systemPrompt) => {
        const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
        const { model, engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
        model.updateConfig({ modelId })
        const bridge = createBrowserBridge(engine)
        vi.stubGlobal('window', { webllmBridge: bridge })
        const page: BrowserPage = {
          async evaluate(run, input) {
            const result = await run(structuredClone(input))
            return result === undefined ? result : JSON.parse(JSON.stringify(result))
          },
        }
        const selected = browser ? new BrowserModel(page, { modelId }) : model
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]
        const options = systemPrompt === undefined ? {} : { systemPrompt }
        try {
          await collect(selected.stream(messages, options))
          const first = create.mock.calls[0]![0].messages[0]!
          if (first.role !== 'system') throw new Error('Expected a system prompt')
          const reference = setup().model
          expect(await selected.countTokens(messages, options)).toBe(
            await reference.countTokens(messages, { systemPrompt: first.content }),
          )
        } finally {
          await bridge.close()
        }
      },
    )

    it('does not count the system prompt again for new messages', async () => {
      const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
      const response = {
        ...chunk(),
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
      const { model, engine } = setup(chunk({ content: 'Done' }, 'stop'), response)
      model.updateConfig({ modelId, contextWindowLimit: 8192 })
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, input) {
          const result = await run(structuredClone(input))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const selected = browser
        ? new BrowserModel(page, { modelId, contextWindowLimit: 8192 })
        : model
      const agent = new Agent({
        model: selected,
        systemPrompt: '',
        contextManager: 'auto',
        printer: false,
      })
      try {
        await collect(agent.stream('First'))
        const events = await collect(agent.stream('Next'))
        const before = events.find((event) => event instanceof BeforeModelCallEvent)
        const reference = setup().model
        const extra = await reference.countTokens([
          new Message({ role: 'user', content: [new TextBlock('Next')] }),
        ])
        expect(before?.projectedInputTokens).toBe(110 + extra)
      } finally {
        await bridge.close()
      }
    })

    it('reports the tool prompt estimate to the Strands agent', async () => {
      const { model, engine, create } = setup(chunk({ content: 'Sunny.' }, 'stop'))
      const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
      const contextWindowLimit = 8192
      model.updateConfig({ modelId, contextWindowLimit })
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, input) {
          const result = await run(structuredClone(input))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const selected = browser ? new BrowserModel(page, { modelId, contextWindowLimit }) : model
      const weather = new FunctionTool({
        name: 'weather',
        description: 'Get the weather.',
        inputSchema: {
          type: 'object',
          properties: { city: { type: 'string' } },
          required: ['city'],
        },
        callback: () => 'Sunny.',
      })
      const utilization = vi.spyOn(selected, 'estimateUtilization')
      const agent = new Agent({
        model: selected,
        tools: [weather],
        contextManager: 'auto',
        printer: false,
      })

      try {
        const events = await collect(agent.stream('What is the weather in Seoul?'))
        const before = events.find((event) => event instanceof BeforeModelCallEvent)
        expect(before?.projectedInputTokens).toBeGreaterThan(200)
        expect(utilization).toHaveBeenCalledWith(before?.projectedInputTokens)
        expect(create.mock.calls[0]?.[0].messages[0]?.content).toContain('"name": "weather"')
      } finally {
        await bridge.close()
      }
    })

    it.each([
      { modelId: models[0]!, min: 130, max: 210 },
      { modelId: models[1]!, min: 120, max: 190 },
      { modelId: models[2]!, min: 200, max: 320 },
    ])('includes the tool instructions sent to $modelId', async ({ modelId, min, max }) => {
      const { model } = setup()
      model.updateConfig({ modelId })
      const page: BrowserPage = {
        async evaluate() {
          throw new Error('Token estimates must not call the browser.')
        },
      }
      const selected = browser ? new BrowserModel(page, { modelId }) : model
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('What is the weather in Seoul?')] }),
      ]
      const tools = [
        {
          name: 'weather',
          description: 'Get the weather.',
          inputSchema: {
            type: 'object' as const,
            properties: { city: { type: 'string' as const } },
            required: ['city'],
          },
        },
      ]

      const estimate = await selected.countTokens(messages, { toolSpecs: tools })
      expect(estimate).toBeGreaterThan(min)
      expect(estimate).toBeLessThan(max)
    })

    it('keeps the SDK estimate for multilingual tool descriptions', async () => {
      const { model } = setup()
      const page: BrowserPage = {
        async evaluate() {
          throw new Error('Token estimates must not call the browser.')
        },
      }
      const messages = [new Message({ role: 'user', content: [new TextBlock('Find my order.')] })]
      for (const description of [
        '고객의 주문 번호를 조회하고 배송 상태를 정확하게 알려주는 도구입니다.',
        '注文番号を調べてお客様に配送状況を正確にお知らせします。',
      ]) {
        const tools = Array.from({ length: 8 }, (_, index) => ({
          name: `order${index}`,
          description,
          inputSchema: {
            type: 'object' as const,
            properties: { order: { type: 'string' as const, description } },
          },
        }))
        model.updateConfig({ modelId: 'test-model' })
        const original = await model.countTokens(messages, { toolSpecs: tools })
        for (const modelId of models) {
          model.updateConfig({ modelId })
          const selected = browser ? new BrowserModel(page, { modelId }) : model
          expect(await selected.countTokens(messages, { toolSpecs: tools })).toBeGreaterThanOrEqual(
            original,
          )
        }
      }
    })

    it.each(models)('%s excludes thoughts omitted after a new question', async (modelId) => {
      const local = setup()
      local.create.mockImplementation(async () => chunks(chunk({ content: 'Done' }, 'stop')))
      local.model.updateConfig({ modelId })
      const bridge = createBrowserBridge(local.engine)
      vi.stubGlobal('window', { webllmBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, input) {
          const result = await run(structuredClone(input))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const model = browser ? new BrowserModel(page, { modelId }) : local.model
      const history = (thought: string) => [
        new Message({ role: 'user', content: [new TextBlock('First question.')] }),
        new Message({
          role: 'assistant',
          content: [new ReasoningBlock({ text: thought }), new TextBlock('Earlier answer.')],
        }),
        new Message({ role: 'user', content: [new TextBlock('Next question.')] }),
      ]
      const short = history('Earlier thought.')
      const long = history('Earlier thought. '.repeat(2000))
      const snapshot = JSON.stringify(long)
      for (const message of long) {
        for (const block of message.content) Object.freeze(block)
        Object.freeze(message.content)
        Object.freeze(message)
      }
      Object.freeze(long)
      try {
        await collect(model.stream(short))
        await collect(model.stream(long))
        expect(local.create.mock.calls[1]?.[0]).toEqual(local.create.mock.calls[0]?.[0])
        expect(JSON.stringify(local.create.mock.calls[0]?.[0])).not.toContain('Earlier thought.')
        expect(await model.countTokens(long)).toBe(await model.countTokens(short))
        expect(JSON.stringify(long)).toBe(snapshot)
      } finally {
        await bridge.close()
      }
    })

    it.each(models)('%s follows its current tool-loop thinking policy', async (modelId) => {
      const { model, engine } = setup()
      model.updateConfig({ modelId })
      const page: BrowserPage = {
        async evaluate() {
          throw new Error('Token estimates must not call the browser.')
        },
      }
      const selected = browser ? new BrowserModel(page, { modelId }) : model
      const history = (thought: string) => [
        new Message({ role: 'user', content: [new TextBlock('Look up A7.')] }),
        new Message({
          role: 'assistant',
          content: [
            new ReasoningBlock({ text: thought }),
            new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: { code: 'A7' } }),
          ],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call',
              status: 'success',
              content: [new TextBlock('42')],
            }),
          ],
        }),
      ]
      const short = await selected.countTokens(history('Current thought.'))
      const long = await selected.countTokens(history('Current thought. '.repeat(100)))
      if (modelId.startsWith('Qwen2.5-')) expect(long).toBe(short)
      else expect(long).toBeGreaterThan(short)
      expect(engine.chat.completions.create).not.toHaveBeenCalled()
    })
  })
}
