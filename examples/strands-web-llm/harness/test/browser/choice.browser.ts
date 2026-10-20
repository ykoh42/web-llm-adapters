import { Message, TextBlock, type ModelStreamEvent } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'

for (const requested of ['weather', 'time']) {
  test(`Hermes named tool choice validates native ${requested} output`, async ({ page }, info) => {
    const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      const native: (string | undefined)[] = []
      const events: ModelStreamEvent[] = []
      const model = new BrowserModel(page, {
        modelId,
        request: { temperature: 0, max_tokens: 4096, seed: 7 },
        onChunk: (chunk) => {
          const choice = chunk.choices[0]
          if (choice && 'delta' in choice)
            for (const call of choice.delta.tool_calls ?? []) native.push(call.function?.name)
        },
      })
      let failure: string | undefined
      try {
        for await (const event of model.stream(
          [
            new Message({
              role: 'user',
              content: [new TextBlock(`Use ${requested} to get the ${requested}.`)],
            }),
          ],
          {
            toolChoice: { tool: { name: 'weather' } },
            toolSpecs: ['weather', 'time'].map((name) => ({
              name,
              description: name === 'weather' ? 'Get the weather.' : 'Get the time.',
              inputSchema: { type: 'object', properties: {} },
            })),
          },
        ))
          events.push(event)
      } catch (error) {
        failure = String(error)
      }
      await writeFile(
        info.outputPath('native-choice.json'),
        JSON.stringify({ native, events, failure }, null, 2),
      )
      expect(native.length).toBeGreaterThan(0)
      const calls = events.filter(
        (event) =>
          event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart',
      )
      if (native.some((name) => name !== 'weather')) {
        expect(failure).toContain('outside tool_choice')
        expect(calls).toEqual([])
      } else {
        expect(failure).toBeUndefined()
        expect(calls.length).toBe(native.length)
      }
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}
