import { test, expect } from './browser.js'
import type {} from './checks.js'

test('a queued timeout leaves the active Worker request and its next request usable', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const context = {
      messages: [{ role: 'user' as const, content: 'What is 3 + 3? Reply briefly.', timestamp: 0 }],
    }
    let entered!: () => void
    let resume!: () => void
    const started = new Promise<void>((resolve) => {
      entered = resolve
    })
    const hold = new Promise<void>((resolve) => {
      resume = resolve
    })
    let observed = false
    let queuedPayloads = 0
    const active = models.complete(model, context, {
      reasoning: 'low',
      onProviderStreamEvent: async () => {
        if (observed) return
        observed = true
        entered()
        await hold
      },
    })
    let queued
    try {
      await started
      queued = await models.complete(model, context, {
        reasoning: 'low',
        timeoutMs: 50,
        onPayload: () => {
          queuedPayloads++
        },
      })
    } finally {
      resume()
    }
    const first = await active
    const recovered = await models.complete(model, context, { reasoning: 'low' })
    return { first, queued, recovered, queuedPayloads }
  })

  expect(result.queued).toMatchObject({
    stopReason: 'aborted',
    errorMessage: expect.stringMatching(/timed out|timeout/i),
    content: [],
  })
  expect(result.queuedPayloads).toBe(0)
  for (const response of [result.first, result.recovered]) {
    expect(response.stopReason).toBe('stop')
    expect(response.usage.totalTokens).toBeGreaterThan(0)
    expect(response.content.some((part) => part.type === 'thinking' && part.thinking)).toBe(true)
    expect(
      response.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    ).toContain('6')
  }
})
