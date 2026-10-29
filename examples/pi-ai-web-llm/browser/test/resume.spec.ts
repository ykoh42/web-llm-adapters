import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import { test, expect } from './browser.js'
import type {} from './checks.js'
import { modelCheck } from '../../../../scripts/release/support.js'
import { checkResume } from '../../../../test/support/resume.js'

test(
  'Pi can resume interrupted native generation',
  modelCheck('engine', 'resume'),
  async ({ page }) => {
    const modelId = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
    const sessionId = crypto.randomUUID()
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    try {
      const stopped = await page.evaluate(async (sessionId) => {
        const { models, model } = await window.piChecks.provider(false, {
          request: {
            max_tokens: 128,
            temperature: 0,
            seed: 7,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId,
                strictPersistence: true,
                checkpointPrompt: false,
              },
            },
          },
        })
        const controller = new AbortController()
        let partial = ''
        const result = await models.complete(
          model,
          { messages: [{ role: 'user', content: 'Explain the seasons in detail.', timestamp: 0 }] },
          {
            signal: controller.signal,
            onProviderStreamEvent: (data) => {
              const text = (data as ChatCompletionChunk).choices[0]?.delta.content
              if (text) partial += text
              if (partial.length >= 8 && !controller.signal.aborted) controller.abort()
            },
          },
        )
        return { partial, stopReason: result.stopReason }
      }, sessionId)
      expect(stopped.stopReason).toBe('aborted')
      await checkResume(page, 'pi', sessionId, stopped.partial)
    } finally {
      await page
        .evaluate(async (id) => {
          const { engine } = await window.piChecks.provider(false)
          return window.piChecks.withEngine(engine, (engine) => engine.deleteResumableSession(id))
        }, sessionId)
        .catch(() => {})
    }
  },
)
