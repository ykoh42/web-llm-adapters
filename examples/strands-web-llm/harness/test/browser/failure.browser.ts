import {
  Message,
  TextBlock,
  ReasoningBlock,
  ToolUseBlock,
  type StreamOptions,
} from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { breakToolStream, expectedOutput } from '../../../../../test/support/stream-failure.js'

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
]) {
  test(`${modelId} preserves thinking and tool bytes through a stream failure`, async ({
    page,
  }) => {
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      const gemma = modelId.startsWith('gemma-')
      await page.evaluate(breakToolStream, {
        app: 'strands' as const,
        call: gemma ? '<|tool_call>' : '<tool_call>',
        after: gemma ? '<channel|>' : '</think>',
      })
      let native = ''
      const settings = sampling(modelId, true)
      const model = new BrowserModel(page, {
        modelId,
        maxTokens: 4096,
        ...settings,
        request: { ...settings.request, seed: 7 },
        onChunk: (chunk) => {
          const choice = chunk.choices[0]
          if (choice) native += 'delta' in choice ? (choice.delta.content ?? '') : choice.text
        },
      })
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
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Use lookup to look up code A7.')] }),
      ]
      let text = ''
      let reasoning = ''
      await expect(
        (async () => {
          for await (const event of model.stream(messages, options)) {
            if (event.type === 'modelContentBlockStartEvent') expect(event.start).toBeUndefined()
            if (event.type === 'modelContentBlockDeltaEvent') {
              if (event.delta.type === 'textDelta') text += event.delta.text
              if (event.delta.type === 'reasoningContentDelta') reasoning += event.delta.text ?? ''
            }
          }
        })(),
      ).rejects.toThrow('Injected native stream failure')
      expect({ text, reasoning }).toEqual(expectedOutput(native, modelId))
      expect(reasoning.length).toBeGreaterThan(0)
      const recovered = []
      for await (const block of model.streamAggregated(messages, options))
        if (block instanceof ToolUseBlock || block instanceof ReasoningBlock) recovered.push(block)
      expect(recovered).toContainEqual(
        expect.objectContaining({ name: 'lookup', input: { code: 'A7' } }),
      )
      expect(recovered.some((block) => block instanceof ReasoningBlock && block.text)).toBe(true)
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}
