import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
]) {
  for (const phase of ['raw callback', 'token callback']) {
    test(`${modelId} cancels at ${phase} and runs the next request`, async ({ page }) => {
      await page.evaluate((id) => window.checks.load(id), modelId)
      const thinking = !modelId.startsWith('Hermes-')
      const controller = new AbortController()
      let callbacks = 0
      let atCancel = 0
      const model = new BrowserChatWebLLM(page, {
        modelId,
        request: { temperature: 0, ...sampling(modelId, thinking), max_tokens: 4096, seed: 7 },
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
      const bound = model.bindTools([
        {
          type: 'function',
          function: {
            name: 'lookup',
            description: 'Look up a code.',
            parameters: {
              type: 'object',
              properties: { code: { type: 'string' } },
              required: ['code'],
            },
          },
        },
      ])
      const output: string[] = []
      let tokenCallbacks = 0
      const onToken = () => {
        tokenCallbacks++
        if (phase === 'token callback') {
          atCancel = callbacks
          controller.abort()
        }
      }
      await expect(
        (async () => {
          for await (const chunk of await bound.stream('Use lookup to look up code A7.', {
            signal: controller.signal,
            callbacks: [{ handleLLMNewToken: onToken }],
          }))
            output.push(chunk.text)
        })(),
      ).rejects.toHaveProperty('name', 'AbortError')
      expect(atCancel).toBeGreaterThan(0)
      expect(callbacks).toBe(atCancel)
      expect(tokenCallbacks).toBe(phase === 'raw callback' ? 0 : 1)
      expect(output).toEqual([])
      const recovered = await bound.invoke('Use lookup to look up code A7.')
      expect(recovered.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
      if (thinking)
        expect(recovered.contentBlocks.some((block) => block.type === 'reasoning')).toBe(true)
    })
  }
}
