import { test, expect } from './browser.js'

test('cancelling while the agent module loads prevents a prompt', async ({ page }) => {
  let waiting = false
  let release!: () => void
  const pending = new Promise<void>((resolve) => (release = resolve))
  await page.route('**/pi/assets/agent-*.js', async (route) => {
    waiting = true
    await pending
    await route.continue()
  })
  try {
    await page.goto('http://127.0.0.1:4173/pi/')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect.poll(() => waiting).toBe(true)
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    release()
    await expect(page.locator('#status')).toHaveText('aborted')
    await expect(page.locator('#answer')).toBeEmpty()
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled()
  } finally {
    release()
  }
})

test('the Pi page preserves tools and thinking after cancelling generation', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/pi/')
  await page.getByRole('button', { name: 'Load model', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled()
  await expect(page.locator('#status')).toHaveText('Ready', { timeout: 15 * 60 * 1000 })
  await expect(page.locator('#thinking')).toBeChecked()
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('stop', { timeout: 120_000 })
  await expect(page.locator('#answer')).toContainText('ORCHID-42')
  await expect(page.locator('#details')).toContainText('"type": "thinking"')
  await expect(page.locator('#details')).toContainText('"type": "toolCall"')

  await page
    .locator('#message')
    .fill('Count from 1 to 1000. Use digits. Do not explain or use tools.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#answer')).toContainText('1', { timeout: 120_000 })
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('aborted', { timeout: 30_000 })

  await page.locator('#message').fill('Look up A7 and tell me the answer.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('stop', { timeout: 120_000 })
  await expect(page.locator('#answer')).toContainText('ORCHID-42')
  await expect(page.locator('#details')).toContainText('"type": "thinking"')
  await expect(page.locator('#details')).toContainText('"type": "toolCall"')
})
