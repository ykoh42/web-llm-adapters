import type { CompletionCreateParamsStreaming } from '@mlc-ai/web-llm'
import { modelCheck } from '../../../../../scripts/release/support.js'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { expect, test } from '../../../fixtures/browser.js'
import type {} from '../../src/main.js'

const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'

declare global {
  interface Window {
    ignoreEosRequests: CompletionCreateParamsStreaming[]
  }
}

test(
  `${modelId} honors ignore_eos for text-only thinking`,
  modelCheck(modelId, 'ignore_eos'),
  async ({ page }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(() => {
      const endpoint = window.checks.engine().completions
      window.ignoreEosRequests = []
      const create = endpoint.create.bind(endpoint)
      endpoint.create = new Proxy(create, {
        apply(target, self, args) {
          window.ignoreEosRequests.push(structuredClone(args[0]))
          return Reflect.apply(target, self, args)
        },
      })
    })
    const model = new BrowserChatWebLLM(page, {
      modelId,
      maxTokens: 64,
      temperature: 0,
      request: { ignore_eos: true, seed: 7, extra_body: { enable_thinking: true } },
    })
    const response = await model.invoke('Say hello briefly.')
    const request = await page.evaluate(() => window.ignoreEosRequests[0]!)

    expect(request.ignore_eos).toBe(true)
    expect(request.prompt).toContain('<|think|>')
    expect(response.response_metadata.finish_reason).toBe('length')
    expect(response.content).not.toBe('')
  },
)
