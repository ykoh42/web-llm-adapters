import { test, expect } from './browser.js'
import type {} from './checks.js'

test('the example UI loads a Worker and runs the Pi tool loop without test globals', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('http://127.0.0.1:4173/pi/')
  expect(await page.evaluate(() => typeof window.piChecks)).toBe('undefined')
  await page.getByRole('button', { name: 'Load model', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('Ready', { timeout: 15 * 60 * 1000 })
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('stop', { timeout: 15 * 60 * 1000 })
  await expect(page.locator('#answer')).toContainText('ORCHID-42')
  await expect(page.locator('#details')).toContainText('toolResult')
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled()
  expect(errors).toEqual([])
})

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
]) {
  test(`${modelId}: Pi Agent Core tools, usage, cancellation and recovery in a Worker`, async ({
    page,
  }, info) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const messages = await page.evaluate(() =>
      window.piChecks.run('Look up A7 and tell me the answer.'),
    )
    await info.attach('messages', {
      body: JSON.stringify(messages, null, 2),
      contentType: 'application/json',
    })
    expect(
      messages.some(
        (message) =>
          message.role === 'toolResult' && message.toolName === 'lookup' && !message.isError,
      ),
    ).toBe(true)
    const final = messages.at(-1)
    expect(final).toMatchObject({ role: 'assistant', stopReason: 'stop' })
    if (final?.role === 'assistant') {
      expect(
        final.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join(''),
      ).toContain('ORCHID-42')
      expect(final.usage.totalTokens).toBeGreaterThan(0)
    }
    if (!modelId.startsWith('Qwen2.5-'))
      expect(
        messages.some(
          (message) =>
            message.role === 'assistant' &&
            message.content.some((part) => part.type === 'thinking' && part.thinking.length > 0),
        ),
      ).toBe(true)
    const recovery = await page.evaluate(() => window.piChecks.cancelAndRecover())
    await info.attach('cancellation', {
      body: JSON.stringify(recovery, null, 2),
      contentType: 'application/json',
    })
    expect(recovery.observed).toBe(true)
    expect(recovery.cancelled.stopReason).toBe('aborted')
    expect(recovery.recovered.message.stopReason).toBe('stop')
    expect(recovery.recovered.events).toContain('text_delta')
    expect(
      recovery.recovered.message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    ).toContain('6')
    expect(errors).toEqual([])
  })
}
