import { Message, TextBlock, ToolUseBlock } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { watchNativeFinish } from '../../../../../test/support/native-stream.js'

test('native tool chunks arrive before WebLLM finishes generating arguments', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    await page.evaluate(watchNativeFinish, 'strands' as const)
    let raw = ''
    let finished: Promise<boolean> | undefined
    const settings = sampling(modelId, true)
    const model = new BrowserModel(page, {
      modelId,
      ...settings,
      maxTokens: 4096,
      request: { ...settings.request, seed: 7 },
      onChunk: (chunk) => {
        const choice = chunk.choices[0]
        if (choice && 'delta' in choice) raw += choice.delta.content ?? ''
        if (
          !finished &&
          raw.includes('</think>') &&
          raw.lastIndexOf('<tool_call>') > raw.lastIndexOf('</think>')
        )
          finished = page.evaluate(
            () => (window as unknown as { nativeStreamDone: boolean }).nativeStreamDone,
          )
      },
    })
    const calls: ToolUseBlock[] = []
    for await (const block of model.streamAggregated(
      [
        new Message({
          role: 'user',
          content: [new TextBlock('Use save_note to save an 80-word paragraph about the ocean.')],
        }),
      ],
      {
        toolSpecs: [
          {
            name: 'save_note',
            description: 'Save a note.',
            inputSchema: {
              type: 'object',
              properties: { content: { type: 'string' } },
              required: ['content'],
            },
          },
        ],
      },
    ))
      if (block instanceof ToolUseBlock) calls.push(block)
    expect(await finished).toBe(false)
    expect(calls).toMatchObject([{ name: 'save_note', input: { content: expect.any(String) } }])
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})
