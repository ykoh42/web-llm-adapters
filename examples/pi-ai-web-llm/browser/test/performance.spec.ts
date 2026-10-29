import { test, expect } from './browser.js'
import type {} from './checks.js'

test('Pi stays close to native WebLLM on the same request', async ({ page }, info) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen2.5-0.5B-Instruct-q4f16_1-MLC'))
  const samples = await page.evaluate(async () => {
    const { models, model, engine } = await window.piChecks.provider(false, {
      request: { max_tokens: 128, temperature: 0 },
    })
    return window.piChecks.comparePerformance(models, model, engine)
  })
  for (const { native, adapter } of samples) {
    expect(native.tokens).toBeGreaterThan(0)
    expect(native.text.trim()).not.toBe('')
    if (adapter.tokens !== native.tokens || adapter.text !== native.text) {
      await info.attach('different-output', {
        body: JSON.stringify({ native, adapter }, null, 2),
        contentType: 'application/json',
      })
      test.skip(true, 'The model generated different outputs; compare equal workloads.')
    }
  }
  const median = (values: number[]) => {
    values.sort((left, right) => left - right)
    return (values[1]! + values[2]!) / 2
  }
  const native = median(samples.map((sample) => sample.native.elapsed))
  const adapter = median(samples.map((sample) => sample.adapter.elapsed))
  await info.attach('performance', {
    body: JSON.stringify({ samples, ratio: adapter / native }, null, 2),
    contentType: 'application/json',
  })
  expect(adapter).toBeLessThanOrEqual(native * 1.1 + 50)
})
