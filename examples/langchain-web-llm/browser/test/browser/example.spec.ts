import { test, expect } from '../../../fixtures/browser.js'

test('the browser example runs tools with thinking on and off', async ({ page }) => {
  await page.getByRole('button', { name: 'Load model', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('Ready', { timeout: 15 * 60 * 1000 })
  await page.evaluate(() => {
    window.exampleThinking = []
    const endpoint = window.checks.engine().chat.completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.exampleThinking.push(args[0].extra_body?.enable_thinking)
        return Reflect.apply(target, self, args)
      },
    })
  })
  for (const enabled of [true, false]) {
    await page.getByLabel('Thinking', { exact: true }).setChecked(enabled)
    await page.locator('#message').fill('Use lookup with code A7, then tell me its answer.')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.locator('#status')).toHaveText('Done', { timeout: 120_000 })
    await expect(page.locator('#answer')).toContainText('42')
    await expect(page.locator('#answer')).not.toContainText('</think>')
    await expect(page.locator('#details')).toContainText('lookup')
    const received = await page.evaluate(() => window.exampleThinking.splice(0))
    expect(received.length).toBeGreaterThan(0)
    expect(received.every((thinking) => thinking === enabled)).toBe(true)
  }
})

declare global {
  interface Window {
    exampleThinking: (boolean | undefined)[]
  }
}
