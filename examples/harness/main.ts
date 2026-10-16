import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { createHarness } from '@strands-agents/harness'
import { chromium } from '@playwright/test'
import { BrowserModel } from './browser-model.ts'

const { values, positionals } = parseArgs({
  options: { read: { type: 'boolean' }, think: { type: 'boolean' } },
  allowPositionals: true,
})
const read = values.read ?? false
const thinking = values.think ?? false
let maxTokens = read ? 512 : 128
if (thinking) maxTokens = 1024
const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
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
    model: new BrowserModel(page, {
      modelId,
      maxTokens,
      temperature: 0,
      ...(thinking && { request: { extra_body: { enable_thinking: true } } }),
    }),
    instructions: 'Answer briefly.',
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
  const result = await agent.invoke(positionals[0] ?? 'Say hello.')
  console.log(
    result.lastMessage.content
      .filter((block) => block.type === 'textBlock')
      .map((block) => block.text)
      .join('\n'),
  )
} finally {
  await browser.close()
}
