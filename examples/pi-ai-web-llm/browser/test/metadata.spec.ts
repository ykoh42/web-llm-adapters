import type { WebLLMChunk } from 'pi-ai-web-llm'
import { test, expect } from './browser.js'
import type {} from './checks.js'

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId}: Pi keeps native response metadata without changing output`, async ({
    page,
  }) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const result = await page.evaluate(async () => {
      const { models, model } = await window.piChecks.provider(true)
      const context = {
        messages: [{ role: 'user' as const, content: 'What is 17 * 23?', timestamp: 0 }],
      }
      const baseline = await models.complete(model, context, { reasoning: 'low' })
      const native: { id?: string; model?: string; reason?: string } = {}
      const response = await models.complete(model, context, {
        reasoning: 'low',
        onProviderStreamEvent: (event) => {
          const chunk = event as WebLLMChunk
          native.id = chunk.id
          native.model = chunk.model
          const reason = chunk.choices[0]?.finish_reason
          if (reason) native.reason = reason
        },
      })
      return { baseline, response, native }
    })

    expect(result.native.id).toBeTruthy()
    expect(result.native.model).toBe(modelId)
    expect(result.native.reason).toBe('stop')
    expect(result.response).toMatchObject({
      responseId: result.native.id,
      responseModel: result.native.model,
      rawStopReason: result.native.reason,
      stopReason: 'stop',
      content: result.baseline.content,
      usage: result.baseline.usage,
    })
    expect(result.response.content.some((part) => part.type === 'thinking' && part.thinking)).toBe(
      true,
    )
  })
}
