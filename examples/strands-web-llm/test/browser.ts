import { test as base, expect } from '@playwright/test'

export { expect }
export const test = base.extend({
  context: async ({ playwright }, use) => {
    const context = await playwright.chromium.launchPersistentContext('.cache/chrome', {
      channel: 'chrome',
      headless: process.env.HEADLESS === '1',
    })
    try {
      await use(context)
    } finally {
      await context.close()
    }
  },
  page: async ({ context }, use) => {
    const page = context.pages()[0] ?? (await context.newPage())
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('http://127.0.0.1:4173')
    await use(page)
    expect(errors).toEqual([])
  },
})
