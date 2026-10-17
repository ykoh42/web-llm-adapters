import { resolve } from 'node:path'
import { test as base, expect } from '../../../fixtures/browser.js'
export { expect }

export const test = base.extend<{ extensionId: string }>({
  context: async ({ playwright }, use) => {
    const paths = [
      resolve('examples/strands-web-llm/browser/dist'),
      resolve('.cache/cos-extension'),
    ].join(',')
    const context = await playwright.chromium.launchPersistentContext(
      '.cache/chromium-extensions',
      {
        channel: 'chromium',
        headless: false,
        ignoreDefaultArgs: ['--disable-extensions'],
        args: [`--disable-extensions-except=${paths}`, `--load-extension=${paths}`],
      },
    )
    try {
      const cos =
        context.serviceWorkers().find((worker) => worker.url().endsWith('/background.js')) ??
        (await context.waitForEvent('serviceworker', {
          predicate: (worker) => worker.url().endsWith('/background.js'),
        }))
      // The test profile must never send the extension's optional telemetry.
      await cos.evaluate(() =>
        chrome.storage.local.set({ telemetryEnabled: false, telemetryNoticeShown: true }),
      )
      expect(await cos.evaluate(() => chrome.storage.local.get('telemetryEnabled'))).toEqual({
        telemetryEnabled: false,
      })
      await use(context)
    } finally {
      await context.close()
    }
  },
  extensionId: async ({ context }, use) => {
    const worker =
      context.serviceWorkers().find((worker) => worker.url().endsWith('/extension-worker.js')) ??
      (await context.waitForEvent('serviceworker', {
        predicate: (worker) => worker.url().endsWith('/extension-worker.js'),
      }))
    await use(new URL(worker.url()).host)
  },
})
