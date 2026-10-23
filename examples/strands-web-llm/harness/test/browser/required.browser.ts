import { Message, TextBlock } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of ['gemma-4-E2B-it-q4f16_1-MLC', 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC']) {
  test(`${modelId} requires its only registered tool`, async ({ page }, info) => {
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      const settings = sampling(modelId, modelId.startsWith('gemma-'))
      const model = new BrowserModel(page, {
        modelId,
        ...settings,
        maxTokens: 4096,
        request: { ...settings.request, seed: 7 },
      })
      const results = []
      for (const toolChoice of [{ tool: { name: 'lookup' } }, { any: {} }]) {
        const stream = model.streamAggregated(
          [new Message({ role: 'user', content: [new TextBlock('Call lookup for code A7.')] })],
          {
            toolChoice,
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
          },
        )
        let next = await stream.next()
        while (!next.done) next = await stream.next()
        const blocks = next.value.message.content
        expect(blocks.filter((block) => block.type === 'toolUseBlock')).toMatchObject([
          { name: 'lookup', input: { code: 'A7' } },
        ])
        results.push(blocks)
      }
      await writeFile(info.outputPath('required-tool.json'), JSON.stringify(results, null, 2))
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}
