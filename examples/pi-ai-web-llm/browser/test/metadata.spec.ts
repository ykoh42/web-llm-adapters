import type { WebLLMChunk } from 'pi-ai-web-llm'
import { nativeOutput } from '../../../../test/support/native-output.js'
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
      const chunks: WebLLMChunk[] = []
      let prompt = ''
      const response = await models.complete(model, context, {
        reasoning: 'low',
        onPayload: (payload) => {
          if (payload && typeof payload === 'object' && 'prompt' in payload)
            prompt = typeof payload.prompt === 'string' ? payload.prompt : ''
        },
        onProviderStreamEvent: (event) => {
          chunks.push(event as WebLLMChunk)
        },
      })
      return { response, chunks, prompt }
    })

    const last = result.chunks.at(-1)!
    const finish = result.chunks
      .flatMap((chunk) => chunk.choices.map((choice) => choice.finish_reason))
      .find(Boolean)
    expect(last.id).toBeTruthy()
    expect(last.model).toBe(modelId)
    expect(last.usage).toBeDefined()
    expect(finish).toBe('stop')
    expect(result.response).toMatchObject({
      responseId: last.id,
      responseModel: last.model,
      rawStopReason: finish,
      stopReason: 'stop',
      usage: { input: last.usage?.prompt_tokens, output: last.usage?.completion_tokens },
    })
    const text = result.response.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('')
    const thinking = result.response.content
      .filter((part) => part.type === 'thinking')
      .map((part) => part.thinking)
      .join('')
    const prefix = modelId.startsWith('Qwen3.5-')
      ? '<think>'
      : result.prompt.endsWith('<|channel>thought\n')
        ? '<|channel>thought\n'
        : ''
    expect(thinking).not.toBe('')
    expect({ text, thinking: [thinking] }).toEqual(nativeOutput(result.chunks, modelId, prefix))
  })
}
