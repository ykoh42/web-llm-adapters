import { chromium, expect, test } from '@playwright/test'

test('Gemma 4 E2B streams through a Strands Agent', async () => {
  const browser = await chromium.launchPersistentContext('.cache/chrome', {
    channel: 'chrome',
    headless: process.env.HEADLESS === '1',
  })
  try {
    const page = await browser.newPage()
    page.on('console', (message) => console.log(message.text()))
    page.on('pageerror', (error) => console.log('Page error:', error.message))
    await page.goto('http://127.0.0.1:4173')
    await page.getByRole('button', { name: 'Load model', exact: true }).click()
    await expect
      .poll(
        async () => {
          const status = await page.getByRole('status').textContent()
          console.log(status)
          return status
        },
        { timeout: 12 * 60 * 1000, intervals: [1000, 5000, 15000] },
      )
      .toBe('Model loaded')
    await page.getByLabel('Message').fill('Reply with exactly this word: ready')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Done', { timeout: 120_000 })
    await expect(page.getByLabel('Response')).not.toBeEmpty()
    console.log('Gemma response:', await page.getByLabel('Response').textContent())
    await page.getByRole('button', { name: 'Unload', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Model unloaded', { timeout: 30_000 })
  } finally {
    await browser.close()
  }
})
