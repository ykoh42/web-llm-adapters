import { sampling } from '../sampling.ts'
import { chromium } from 'playwright'
import { fileURLToPath } from 'node:url'
import { createDeepAgent } from 'deepagents'
import { BrowserChatWebLLM } from 'langchain-web-llm'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const cache = fileURLToPath(new URL('../../../.cache/langchain-example', import.meta.url))
const context = await chromium.launchPersistentContext(cache, {
  channel: 'chrome',
  headless: false,
})
try {
  const page = context.pages()[0] ?? (await context.newPage())
  await page.goto('http://127.0.0.1:4175/langchain/')
  await page.evaluate(async (id) => {
    await window.checks.load(id)
  }, modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: await page.evaluate(() => window.checks.model().profile),
    thinkingPrefilled: true,
    request: sampling(modelId, true),
  })
  const agent = createDeepAgent({ model })
  const result = await agent.invoke({
    messages: [{ role: 'user', content: 'Write a short greeting to /greeting.txt, then read it.' }],
  })
  console.log(result.messages.at(-1)?.text)
} finally {
  await context.close()
}
