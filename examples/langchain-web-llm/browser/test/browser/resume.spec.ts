import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { modelCheck } from '../../../../../scripts/release/support.js'
import { checkResume } from '../../../../../test/support/resume.js'

test(
  'LangChain can resume interrupted native generation',
  modelCheck('engine', 'resume'),
  async ({ page }) => {
    const modelId = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
    const sessionId = crypto.randomUUID()
    await page.evaluate((id) => window.checks.load(id), modelId)
    const controller = new AbortController()
    let partial = ''
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: {
        max_tokens: 128,
        temperature: 0,
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
      await expect(
        model.invoke('Explain the seasons in detail.', { signal: controller.signal }),
      ).rejects.toHaveProperty('name', 'AbortError')
      await checkResume(page, 'langchain', sessionId, partial)
    } finally {
      await page
        .evaluate(
          (id) =>
            window.webllmLangChainBridge!.withEngine((engine) => engine.deleteResumableSession(id)),
          sessionId,
        )
        .catch(() => {})
    }
  },
)
