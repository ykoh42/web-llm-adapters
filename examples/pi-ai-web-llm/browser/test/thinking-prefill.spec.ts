import type {
  ChatCompletionChunk,
  ChatCompletionRequestStreaming,
  ChatOptions,
} from '@mlc-ai/web-llm'
import { test, expect } from './browser.js'
import type {} from './checks.js'

test('Pi reads thinking prefilled by the loaded Qwen template', async ({ page, request }, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  const response = await request.get(
    `https://huggingface.co/mlc-ai/${modelId}/resolve/main/mlc-chat-config.json`,
  )
  expect(response.ok()).toBe(true)
  const config: ChatOptions = await response.json()
  expect(config.conv_template).toBeDefined()
  const template = { ...config.conv_template!, role_empty_sep: '\n<think>\n' }
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate((id) => window.piChecks.load(id), modelId)
  const result = await page.evaluate(
    async ({ modelId, template }) => {
      const { models, model, engine } = await window.piChecks.provider(true, {
        thinkingPrefilled: true,
      })
      await window.piChecks.withEngine(engine, (engine) =>
        engine.reload(modelId, { context_window_size: 32768, conv_template: template }),
      )
      let payload: ChatCompletionRequestStreaming | undefined
      let raw = ''
      const message = await models.completeSimple(
        model,
        { messages: [{ role: 'user', content: 'What is 17 * 23?', timestamp: 0 }] },
        {
          reasoning: 'high',
          onPayload(request) {
            payload = structuredClone(request) as ChatCompletionRequestStreaming
          },
          onProviderStreamEvent(event) {
            const chunk = event as ChatCompletionChunk
            raw += chunk.choices[0]?.delta.content ?? ''
          },
        },
      )
      if (!payload) throw new Error('Missing native request.')
      return { raw, payload, message }
    },
    { modelId, template },
  )
  await info.attach('prefilled-thinking', {
    body: JSON.stringify(result),
    contentType: 'application/json',
  })
  const reasoning = result.message.content
    .filter((part) => part.type === 'thinking')
    .map((part) => part.thinking)
    .join('')
  const text = result.message.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('')
  expect(result.payload.extra_body?.enable_thinking).toBe(true)
  const end = result.raw.indexOf('</think>')
  expect(end).toBeGreaterThan(0)
  expect(reasoning.trim()).not.toBe('')
  expect(reasoning).toBe(result.raw.slice(0, end))
  expect(text).toBe(result.raw.slice(end + '</think>'.length))
  expect(text).toContain('391')
  expect(result.message.stopReason).toBe('stop')
})
