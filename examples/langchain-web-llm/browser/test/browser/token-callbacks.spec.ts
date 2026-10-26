import { BaseCallbackHandler } from '@langchain/core/callbacks/base'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId}: a callback enabled during generation receives the answer`, async ({ page }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const tokens: string[] = []
    const handler = BaseCallbackHandler.fromMethods({
      handleLLMNewToken(token) {
        tokens.push(token)
      },
    })
    handler.ignoreLLM = true
    handler.awaitHandlers = true
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: () => {
        handler.ignoreLLM = false
      },
    })

    const result = await model.invoke('What is 3 + 3? Reply briefly.', { callbacks: [handler] })

    expect(tokens.length).toBeGreaterThan(0)
    expect(tokens.join('')).toBe(result.text)
    expect(result.text).toContain('6')
  })

  test(`${modelId}: invoke cancels while a token callback is pending`, async ({ page }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    })
    const controller = new AbortController()
    let finish!: () => void
    const hold = new Promise<void>((resolve) => (finish = resolve))
    let callbacks = 0
    const handler = BaseCallbackHandler.fromMethods({
      handleLLMNewToken() {
        callbacks++
        return hold
      },
    })
    handler.awaitHandlers = true
    const pending = model.invoke('Explain the history of mathematics in detail.', {
      signal: controller.signal,
      callbacks: [handler],
    })
    let settled = false
    void pending.then(
      () => (settled = true),
      () => (settled = true),
    )
    try {
      await expect.poll(() => callbacks, { timeout: 60_000 }).toBe(1)
      expect(settled).toBe(false)
      controller.abort()
      await expect.poll(() => settled).toBe(true)
      await expect(pending).rejects.toHaveProperty('name', 'AbortError')
      expect((await model.invoke('What is 3 + 3? Reply briefly.')).text).toContain('6')
      expect(callbacks).toBe(1)
    } finally {
      finish()
      await pending.catch(() => {})
    }
  })
}
