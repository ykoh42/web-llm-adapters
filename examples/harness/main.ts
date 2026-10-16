import { fileURLToPath } from 'node:url'
import { createHarness } from '@strands-agents/harness'
import { AfterToolsEvent } from '@strands-agents/sdk'
import { chromium } from '@playwright/test'
import { BrowserModel } from './browser-model.ts'

const read = process.argv[2] === '--read'
const modelId = read ? 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' : 'gemma-4-E2B-it-q4f16_1-MLC'
const browser = await chromium.launchPersistentContext(
  fileURLToPath(new URL('../../.cache/chrome', import.meta.url)),
  {
    channel: 'chrome',
    headless: false,
  },
)
try {
  const page = browser.pages()[0] ?? (await browser.newPage())
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  const agent = await createHarness({
    model: new BrowserModel(page, { modelId, maxTokens: read ? 256 : 128, temperature: 0 }),
    systemPrompt: read ? '' : 'Answer briefly.',
    builtinTools: read ? ['read'] : [],
    builtinPlugins: [],
    session: false,
    skills: false,
    memory: false,
    contextManager: false,
    caching: false,
    backgroundTasks: false,
    printer: false,
  })
  if (read)
    agent.addHook(AfterToolsEvent, (event) => {
      console.log(JSON.stringify(event.message, null, 2))
      event.endTurn = 'Read finished.'
    })
  console.log(String(await agent.invoke(process.argv[read ? 3 : 2] ?? 'Say hello.')))
} finally {
  await browser.close()
}
