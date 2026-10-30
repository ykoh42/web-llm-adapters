import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { recordThinking } from '../../../../../scripts/release/support.js'

const lookupSchema = z.object({ code: z.string() })
const lookup = tool(({ code }) => `Result for ${code}: ORCHID-42`, {
  name: 'lookup',
  description: 'Look up the answer for a code.',
  schema: lookupSchema,
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(
    `${modelId} streams thinking and a tool call through native LangChain events`,
    { tag: '@release' },
    async ({ page }) => {
      await page.evaluate((id) => window.checks.load(id), modelId)
      await page.evaluate(() => {
        const engine = window.checks.engine()
        window.eventRequests = []
        for (const endpoint of [engine.chat.completions, engine.completions]) {
          endpoint.create = new Proxy(endpoint.create, {
            apply(target, self, args) {
              window.eventRequests.push(structuredClone(args[0]))
              return Reflect.apply(target, self, args)
            },
          })
        }
      })
      const raw = new Map<string, string>()
      const model = new BrowserChatWebLLM(page, {
        modelId,
        thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
        request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
        onChunk: (chunk) => {
          const choice = chunk.choices[0]
          const text = choice && ('delta' in choice ? choice.delta.content : choice.text)
          raw.set(chunk.id, (raw.get(chunk.id) ?? '') + (text ?? ''))
        },
      })
      const input = new HumanMessage('Use lookup to look up code A7, then report its result.')
      const stream = model.streamEvents([input], { tools: [lookup] })
      const [thought, calls, usage, message] = await Promise.all([
        stream.reasoning,
        stream.toolCalls,
        stream.usage,
        stream,
      ])
      const native = [...raw.values()][0]!
      const match = modelId.startsWith('gemma-')
        ? /<\|channel>thought\n([\s\S]*?)<channel\|>/.exec(native)
        : /^([\s\S]*?)<\/think>/.exec(native)
      expect(match?.[1]).toBeTruthy()
      expect(thought).toBe(match![1])
      expect(calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
      expect(message.tool_calls).toEqual(calls)
      expect(usage?.output_tokens).toBeGreaterThan(0)

      const call = calls[0]!
      const result = await lookup.invoke(lookupSchema.parse(call.args))
      if (typeof result !== 'string') throw new Error('Expected a text tool result.')
      const final = model.streamEvents(
        [input, message, new ToolMessage({ tool_call_id: call.id!, content: result })],
        { tools: [lookup] },
      )
      expect(await final.text).toContain('ORCHID-42')
      expect((await final).tool_calls).toEqual([])
      const requests = await page.evaluate(() => window.eventRequests)
      expect(requests).toHaveLength(2)
      const recorded: { annotations: { type: string; description?: string }[] } = {
        annotations: [],
      }
      recordThinking(recorded, modelId, [thought], requests)
      expect(recorded.annotations).toEqual([
        { type: 'supports', description: `${modelId}/thinking` },
      ])
    },
  )
}

test(
  'cancels native LangChain events and reuses the engine',
  { tag: '@release' },
  async ({ page }) => {
    const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
    await page.evaluate((id) => window.checks.load(id), modelId)
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: true,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    })
    const controller = new AbortController()
    const stream = model.streamEvents('Explain the history of mathematics in detail.', {
      signal: controller.signal,
    })
    const pending = (async () => {
      for await (const event of stream) {
        if (event.event === 'content-block-delta' && event.delta.type === 'reasoning-delta')
          controller.abort()
      }
    })()

    await expect(pending).rejects.toHaveProperty('name', 'AbortError')
    expect((await model.invoke('What is 3 + 3? Reply briefly.')).text).toContain('6')
  },
)

declare global {
  interface Window {
    eventRequests: (ChatCompletionRequestStreaming | CompletionCreateParamsStreaming)[]
  }
}
