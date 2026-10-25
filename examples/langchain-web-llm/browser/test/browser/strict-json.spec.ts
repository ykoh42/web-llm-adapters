import { modelCheck } from '../../../../../scripts/release/support.js'
import { writeFile } from 'node:fs/promises'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

interface Answer {
  answer: number
  code: string
  values: number[]
}

const schema = {
  type: 'object',
  properties: {
    answer: { type: 'integer', const: 42 },
    code: { type: 'string', enum: ['A7'] },
    values: { type: 'array', items: { type: 'integer', enum: [1, 2] }, minItems: 2, maxItems: 2 },
  },
  required: ['answer', 'code', 'values'],
  additionalProperties: false,
}

declare global {
  interface Window {
    strictJSONRequests: ChatCompletionRequestStreaming[]
  }
}

test('strict JSON uses native schema constraints in local and browser models', async ({
  page,
}, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  const prompt = 'Return JSON with answer 0, code B8, values [9, 9, 9], and extra true.'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    window.strictJSONRequests = []
    const endpoint = window.checks.engine().chat.completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.strictJSONRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
  const results = []
  for (const strict of [undefined, false, true]) {
    const config = {
      method: 'jsonSchema',
      ...(strict !== undefined && { strict }),
      includeRaw: true as const,
    }
    const local = await page.evaluate(
      async ({ schema, prompt, config }) => {
        const response = await window.checks
          .model()
          .withStructuredOutput<Answer>(schema, config)
          .invoke(prompt)
        return {
          parsed: response.parsed,
          text: response.raw.text,
          finish: response.raw.response_metadata.finish_reason,
        }
      },
      { schema, prompt, config },
    )
    results.push({ client: 'local', strict, ...local })
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    })
    const browser = await model.withStructuredOutput<Answer>(schema, config).invoke(prompt)
    results.push({
      client: 'browser',
      strict,
      parsed: browser.parsed,
      text: browser.raw.text,
      finish: browser.raw.response_metadata.finish_reason,
    })
  }
  const requests = await page.evaluate(() => window.strictJSONRequests)
  const native = await page.evaluate(() =>
    window.webllmLangChainBridge!.withEngine(async (engine) => {
      await engine.resetChat()
      let text = ''
      for await (const chunk of await engine.chat.completions.create(window.strictJSONRequests[0]!))
        text += chunk.choices[0]?.delta.content ?? ''
      return text
    }),
  )
  const evidence = info.outputPath('strict-json.json')
  await writeFile(evidence, JSON.stringify({ results, requests, native }, null, 2))
  await info.attach('strict-json', { path: evidence, contentType: 'application/json' })
  expect(requests).toHaveLength(6)
  for (const request of requests) {
    expect(request).toEqual(requests[0])
    expect(request.messages).toEqual([{ role: 'user', content: prompt }])
    expect(request.response_format).toEqual({ type: 'json_object', schema: JSON.stringify(schema) })
    expect(request.extra_body?.enable_thinking).toBe(true)
  }
  for (const result of results) {
    expect(result.text).toBe(native)
    expect(result.parsed).toEqual(JSON.parse(native))
    expect(result.parsed).toMatchObject({ answer: 42, code: 'A7', values: expect.any(Array) })
    expect(Object.keys(result.parsed).sort()).toEqual(['answer', 'code', 'values'])
    expect(result.parsed.values).toHaveLength(2)
    expect(result.parsed.values.every((value: number) => value === 1 || value === 2)).toBe(true)
    expect(result.finish).toBe('stop')
  }
})

test(
  'structured JSON parses with thinking off and preserves literal tags',
  modelCheck('Qwen3.5-4B-q4f16_1-MLC'),
  async ({ page }) => {
    const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
    await page.evaluate((id) => window.checks.load(id), modelId)
    const text = '<think>literal</think>'
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { max_tokens: 4096, extra_body: { enable_thinking: false } },
    })
    const result = await model
      .withStructuredOutput<{ text: string }>(
        {
          type: 'object',
          properties: { text: { const: text } },
          required: ['text'],
          additionalProperties: false,
        },
        { method: 'jsonSchema', strict: true, includeRaw: true },
      )
      .invoke('Return the requested JSON.')
    expect(result.parsed).toEqual({ text })
    expect(JSON.parse(result.raw.text)).toEqual(result.parsed)
  },
)
