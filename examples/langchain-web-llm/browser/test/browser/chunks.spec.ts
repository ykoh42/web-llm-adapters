import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { watchNativeFinish } from '../../../../../test/support/native-stream.js'

test('native tool chunks arrive before WebLLM finishes generating arguments', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(watchNativeFinish, 'langchain' as const)
  let raw = ''
  let finished: Promise<boolean> | undefined
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
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
  const result = await model
    .bindTools([
      {
        type: 'function',
        function: {
          name: 'save_note',
          description: 'Save a note.',
          parameters: {
            type: 'object',
            properties: { content: { type: 'string' } },
            required: ['content'],
          },
        },
      },
    ])
    .invoke('Use save_note to save an 80-word paragraph about the ocean.')
  expect(await finished).toBe(false)
  expect(result.tool_calls).toMatchObject([
    { name: 'save_note', args: { content: expect.any(String) } },
  ])
})
