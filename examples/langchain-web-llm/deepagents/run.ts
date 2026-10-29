import { sampling } from '../sampling.ts'
import { chromium } from 'playwright'
import { fileURLToPath } from 'node:url'
import { createDeepAgent } from 'deepagents'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { prebuiltAppConfig } from '@mlc-ai/web-llm'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const cache = fileURLToPath(new URL('../../../.cache/langchain-example', import.meta.url))
const context = await chromium.launchPersistentContext(cache, {
  channel: 'chrome',
  headless: false,
})
try {
  const page = context.pages()[0] ?? (await context.newPage())
  await page.goto('http://127.0.0.1:4175/langchain/')
  await page.getByLabel('Model').selectOption(modelId)
  await page.getByRole('button', { name: 'Load model' }).click()
  await page.waitForFunction(
    () => !document.querySelector<HTMLButtonElement>('#load')?.disabled,
    undefined,
    { timeout: 15 * 60 * 1000 },
  )
  const status = await page.locator('#status').textContent()
  if (status !== 'Ready') throw new Error(status ?? 'Failed to load the model.')
  const contextWindow = prebuiltAppConfig.model_list.find((model) => model.model_id === modelId)
    ?.overrides?.context_window_size
  const model = new BrowserChatWebLLM(page, {
    modelId,
    ...(contextWindow !== undefined && { profile: { maxInputTokens: contextWindow } }),
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
