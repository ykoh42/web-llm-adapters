import {
  Message,
  TextBlock,
  JsonBlock,
  ReasoningBlock,
  ToolUseBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import { BrowserModel, type BrowserPage } from 'strands-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming

for (const { modelId, mixed } of [
  { modelId: 'Qwen3.5-4B-q4f16_1-MLC', mixed: false },
  { modelId: 'gemma-4-E2B-it-q4f16_1-MLC', mixed: false },
  { modelId: 'gemma-4-E2B-it-q4f16_1-MLC', mixed: true },
]) {
  test(`${modelId} direct and browser JSON history agree (mixed=${mixed})`, async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      await page.evaluate(() =>
        window.webllmBridge.withEngine((engine) => {
          Object.assign(window, { nativeRequests: [] })
          for (const endpoint of [engine.chat.completions, engine.completions]) {
            Object.assign(endpoint, {
              create: new Proxy(endpoint.create, {
                apply(target, self, args) {
                  ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
                    structuredClone(args[0]),
                  )
                  return Reflect.apply(target, self, args)
                },
              }),
            })
          }
        }),
      )
      const settings = sampling(modelId, true)
      const config = {
        modelId,
        maxTokens: 4096,
        ...settings,
        request: { ...settings.request, seed: 7 },
      }
      const code = 'code-' + crypto.randomUUID().slice(0, 8)
      const input = { code, empty: '', zero: 0, disabled: false, nullable: null }
      Object.assign(input, { optional: undefined, date: new Date('2026-10-20T00:00:00.000Z') })
      const rows = [1]
      rows.length = 3
      const history = [
        new Message({
          role: 'user',
          content: [new TextBlock('Use lookup, then report the exact code it returns.')],
          metadata: { custom: { private: 'session-only', score: Infinity } },
        }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call',
              status: 'success',
              content: [
                new JsonBlock({ json: { code, rows } }),
                ...(mixed ? [new TextBlock('Report the code.')] : []),
              ],
            }),
          ],
        }),
      ]
      const transport: BrowserPage = page
      const direct = await transport.evaluate(
        async (request) => {
          const bridge = window.webllmBridge
          const id = await bridge.start(request)
          let text = ''
          let reasoning = ''
          try {
            while (true) {
              const next = await bridge.next(id)
              if ('error' in next) throw new Error(next.error.message)
              if (next.done) return { text, reasoning }
              if ('native' in next) continue
              const event = next.value
              if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
                text += event.delta.text
              if (
                event.type === 'modelContentBlockDeltaEvent' &&
                event.delta.type === 'reasoningContentDelta'
              )
                reasoning += event.delta.text ?? ''
            }
          } finally {
            await bridge.stop(id)
          }
        },
        { config, messages: history.map((message) => message.toJSON()), options: {} },
      )
      await page.evaluate(() => window.webllmBridge.withEngine((engine) => engine.resetChat()))
      let output = ''
      let reasoning = ''
      for await (const event of new BrowserModel(page, config).streamAggregated(history)) {
        if (event instanceof TextBlock) output += event.text
        if (event instanceof ReasoningBlock) reasoning += event.text ?? ''
      }
      const requests = await page.evaluate(
        () => (window as unknown as { nativeRequests: NativeRequest[] }).nativeRequests,
      )
      expect(requests).toHaveLength(2)
      expect(requests[1]).toEqual(requests[0])
      const request = requests[0]!
      if ('messages' in request) {
        const assistant = request.messages.find((message) => message.role === 'assistant')?.content
        expect(assistant).not.toContain('<parameter=optional>')
        expect(assistant).toContain('<parameter=date>\n2026-10-20T00:00:00.000Z\n</parameter>')
        expect(request.messages.find((message) => message.role === 'user')?.content).not.toContain(
          'session-only',
        )
      } else expect(request.prompt).toContain('rows:[1,null,null]')
      expect(direct.text).toContain(code)
      expect(direct.reasoning).not.toBe('')
      expect(output).toContain(code)
      expect(reasoning).not.toBe('')
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}
