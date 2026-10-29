import { Message, TextBlock } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { modelCheck } from '../../../../../scripts/release/support.js'
import { checkResume } from '../../../../../test/support/resume.js'

test(
  'Strands can resume interrupted native generation',
  modelCheck('engine', 'resume'),
  async ({ page }) => {
    const modelId = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
    const sessionId = crypto.randomUUID()
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id, 8192), modelId)
    const controller = new AbortController()
    let partial = ''
    const model = new BrowserModel(page, {
      modelId,
      maxTokens: 128,
      temperature: 0,
      request: {
        seed: 7,
        extra_body: {
          resumable: { enabled: true, sessionId, strictPersistence: true, checkpointPrompt: false },
        },
      },
      onChunk: (chunk) => {
        const choice = chunk.choices[0]
        const text = choice && 'delta' in choice ? choice.delta.content : undefined
        if (text) partial += text
        if (partial.length >= 8 && !controller.signal.aborted) controller.abort()
      },
    })
    try {
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Explain the seasons in detail.')] }),
      ]
      await expect(
        (async () => {
          for await (const event of model.stream(messages, { cancelSignal: controller.signal }))
            if (event.type === 'modelMessageStopEvent') break
        })(),
      ).rejects.toHaveProperty('name', 'AbortError')
      await checkResume(page, 'strands', sessionId, partial)
    } finally {
      await page
        .evaluate(
          (id) => window.webllmBridge.withEngine((engine) => engine.deleteResumableSession(id)),
          sessionId,
        )
        .catch(() => {})
      await page.evaluate(() => window.harness.close())
    }
  },
)
