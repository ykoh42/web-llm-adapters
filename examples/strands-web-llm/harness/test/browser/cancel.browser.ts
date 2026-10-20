import {
  Message,
  TextBlock,
  ToolUseBlock,
  ReasoningBlock,
  type StreamOptions,
} from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
]) {
  for (const phase of ['raw callback', 'content start']) {
    test(`${modelId} cancels at ${phase} and runs the next request`, async ({ page }) => {
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id), modelId)
      try {
        const thinking = !modelId.startsWith('Hermes-')
        const controller = new AbortController()
        const settings = sampling(modelId, thinking)
        let callbacks = 0
        let atCancel = 0
        const model = new BrowserModel(page, {
          modelId,
          maxTokens: 4096,
          temperature: 0,
          ...settings,
          request: { ...settings.request, seed: 7 },
          onChunk: (chunk) => {
            callbacks++
            const choice = chunk.choices[0]
            const text = choice && ('delta' in choice ? choice.delta.content : choice.text)
            if (phase === 'raw callback' && text && !controller.signal.aborted) {
              atCancel = callbacks
              controller.abort()
            }
          },
        })
        const messages = [
          new Message({ role: 'user', content: [new TextBlock('Use lookup to look up code A7.')] }),
        ]
        const options: StreamOptions = {
          toolSpecs: [
            {
              name: 'lookup',
              description: 'Look up a code.',
              inputSchema: {
                type: 'object',
                properties: { code: { type: 'string' } },
                required: ['code'],
              },
            },
          ],
        }
        const output: unknown[] = []
        await expect(
          (async () => {
            for await (const event of model.stream(messages, {
              ...options,
              cancelSignal: controller.signal,
            })) {
              if (phase === 'content start' && event.type === 'modelContentBlockStartEvent') {
                atCancel = callbacks
                controller.abort()
              }
              if (event.type === 'modelContentBlockDeltaEvent') output.push(event)
            }
          })(),
        ).rejects.toHaveProperty('name', 'AbortError')
        expect(atCancel).toBeGreaterThan(0)
        expect(callbacks).toBe(atCancel)
        expect(output).toEqual([])
        const recovered = []
        for await (const block of model.streamAggregated(messages, options))
          if (block instanceof ToolUseBlock || block instanceof ReasoningBlock)
            recovered.push(block)
        expect(recovered).toContainEqual(
          expect.objectContaining({ name: 'lookup', input: { code: 'A7' } }),
        )
        if (thinking)
          expect(recovered.some((block) => block instanceof ReasoningBlock && block.text)).toBe(
            true,
          )
      } finally {
        await page.evaluate(() => window.harness.close())
      }
    })
  }
}
