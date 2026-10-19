import { test as base, expect } from '@playwright/test'
import type {} from '../browser/src/main.js'

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
    page.on('requestfailed', (request) =>
      console.error('Request failed:', request.url().split('?')[0], request.failure()?.errorText),
    )
    page.on('response', (response) => {
      if (response.status() >= 400)
        console.error('HTTP', response.status(), response.url().split('?')[0])
    })
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('http://127.0.0.1:4173/langchain/')
    await use(page)
    expect(errors).toEqual([])
  },
})

export { expect }
