import { chromium, expect, test } from '@playwright/test'

test('Gemma 4 E2B streams through a Strands Agent', async () => {
  const browser = await chromium.launchPersistentContext('.cache/chrome', {
    channel: 'chrome',
    headless: process.env.HEADLESS === '1',
  })
  try {
    const page = await browser.newPage()
    const errors: string[] = []
    page.on('console', (message) => console.log(message.text()))
    page.on('pageerror', (error) => errors.push(error.message))
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
    const output = page.getByLabel('Response')
    await output.evaluate((element) => {
      let updates = 0
      new MutationObserver(() => {
        const status = document.querySelector('[role="status"]')?.textContent
        if (element.textContent && status === 'Running…') {
          element.setAttribute('data-updates', String(++updates))
        }
      }).observe(element, { childList: true, characterData: true, subtree: true })
    })
    await page
      .getByLabel('Message')
      .fill('Count from 1 to 20. Separate numbers with spaces. Do not explain.')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Done', { timeout: 120_000 })
    await expect(output).not.toBeEmpty()
    expect(Number(await output.getAttribute('data-updates'))).toBeGreaterThan(1)
    expect(errors).toEqual([])
    console.log('Gemma response:', await output.textContent())
    console.log('Visible stream updates:', await output.getAttribute('data-updates'))
    await page.getByRole('button', { name: 'Unload', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Model unloaded', { timeout: 30_000 })
  } finally {
    await browser.close()
  }
})
