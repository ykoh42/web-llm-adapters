import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { createHarness } from '@strands-agents/harness'
import { chromium } from '@playwright/test'
import { BrowserModel } from './browser-model.ts'
import { useTool } from './tool-choice.ts'
import { createMemory } from './memory.ts'

const { values, positionals } = parseArgs({
  options: { think: { type: 'boolean' }, tool: { type: 'string' } },
  allowPositionals: true,
})
const thinking = values.think ?? false
const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
const auxId = 'Qwen3-0.6B-q4f16_1-MLC'
const contextWindow = 16384
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
  await page.evaluate(
    ({ modelIds, contextWindow }) => window.harness.open(modelIds, contextWindow),
    {
      modelIds: [modelId, auxId],
      contextWindow,
    },
  )
  const config = { modelId, contextWindowLimit: contextWindow }
  const auxModel = new BrowserModel(page, {
    ...config,
    modelId: auxId,
    maxTokens: 512,
    temperature: 0,
    request: { extra_body: { enable_thinking: false } },
  })
  const agent = await createHarness({
    model: new BrowserModel(page, {
      ...config,
      maxTokens: 4096,
      temperature: 1,
      topP: 0.95,
      request: { repetition_penalty: 1.05, extra_body: { enable_thinking: thinking } },
    }),
    auxModel,
    memory: { stores: [createMemory(page, modelId, contextWindow)] },
    instructions: 'Answer briefly.',
    builtinTools: { read: { media: false }, web_fetch: { model: auxModel } },
    printer: false,
  })
  if (values.tool) useTool(agent, values.tool)
  try {
    const result = await agent.invoke(positionals[0] ?? 'Say hello.')
    console.log(
      result.lastMessage.content
        .filter((block) => block.type === 'textBlock')
        .map((block) => block.text)
        .join('\n'),
    )
  } finally {
    await agent.shutdown()
  }
} finally {
  await browser.close()
}
