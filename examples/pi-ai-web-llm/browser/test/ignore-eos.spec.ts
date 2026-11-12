import type { CompletionCreateParamsStreaming } from '@mlc-ai/web-llm'
import { modelCheck } from '../../../../scripts/release/support.js'
import { expect, test } from './browser.js'
import type {} from './checks.js'

const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'

test(
  `${modelId} honors ignore_eos for text-only thinking`,
  modelCheck(modelId, 'ignore_eos'),
  async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const evidence = await page.evaluate(async () => {
      const { models, model, engine } = await window.piChecks.provider(true, {
        request: {
          ignore_eos: true,
          max_tokens: 256,
          seed: 7,
          extra_body: { enable_thinking: true },
        },
      })
      let request: CompletionCreateParamsStreaming | undefined
      const endpoint = engine.completions
      const create = endpoint.create.bind(endpoint)
      endpoint.create = new Proxy(create, {
        apply(target, self, args) {
          request = structuredClone(args[0])
          return Reflect.apply(target, self, args)
        },
      })
      const message = await models.complete(
        model,
        { messages: [{ role: 'user', content: 'Say hello briefly.', timestamp: 0 }] },
        {},
      )
      return { message, request }
    })

    expect(evidence.request?.ignore_eos).toBe(true)
    expect(evidence.request?.prompt).toContain('<|think|>')
    expect(evidence.message.stopReason).toBe('length')
    expect(evidence.message.content.some((part) => part.type === 'text' && part.text.trim())).toBe(
      true,
    )
  },
)
