import { chromium, type FullConfig } from '@playwright/test'
import { checkClean } from './source.ts'

export default async function setup(config: FullConfig) {
  checkClean()
  if (config.shard || config.projects.some((project) => project.retries))
    throw new Error('Release model checks must run together without retries.')
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: process.env.HEADLESS === '1',
  })
  try {
    const page = await browser.newPage()
    await page.goto('http://127.0.0.1:4173')
    const gpu = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter || adapter.info.isFallbackAdapter)
        throw new Error('A hardware WebGPU adapter is required.')
      const { vendor, architecture, device, description } = adapter.info
      return { vendor, architecture, device, description }
    })
    process.env.WEB_LLM_RELEASE_ENVIRONMENT = JSON.stringify({
      platform: process.platform,
      arch: process.arch,
      node: process.version,
      browser: browser.version(),
      gpu,
    })
  } finally {
    await browser.close()
  }
}
