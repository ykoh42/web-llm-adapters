import { test as base, expect } from '@playwright/test'

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
})

export { expect }
