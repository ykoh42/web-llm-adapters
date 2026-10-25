import { chatOptions } from '../../../chat-options.js'
import { ContextOverflowError, getRetryable } from '@langchain/core/errors'
import { HumanMessage, AIMessage } from '@langchain/core/messages'
import { createDeepAgent, createSummarizationMiddleware, StateBackend } from 'deepagents'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const worker of [false, true]) {
  test(`native context overflow keeps its LangChain type with worker ${worker}`, async ({
    page,
  }) => {
    const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
    await page.evaluate(({ modelId, worker }) => window.checks.load(modelId, worker), {
      modelId,
      worker,
    })
    await page.evaluate(
      (modelId) => window.checks.engine().reload(modelId, { context_window_size: 1024 }),
      modelId,
    )
    const model = new BrowserChatWebLLM(page, {
      modelId,
      profile: { maxInputTokens: 1024 },
      request: { temperature: 0, max_tokens: 128 },
      maxRetries: 1,
    })
    const error = await model.invoke('word '.repeat(3000)).catch((error: unknown) => error)
    expect(ContextOverflowError.isInstance(error)).toBe(true)
    expect(getRetryable(error)).toBe(false)
    expect(error).toHaveProperty('cause')
    expect((await model.invoke('Reply with the word ready.')).text.toLowerCase()).toContain('ready')
  })
}

test('Deep Agents summarizes and recovers after a real WebLLM overflow', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(
    async ({ id, options }) => {
      await window.checks.engine().reload(id, { ...options, context_window_size: 16384 })
      Object.assign(window, { contextOverflows: 0 })
      const bridge = window.webllmLangChainBridge!
      const next = bridge.next.bind(bridge)
      bridge.next = async (id) => {
        const result = await next(id)
        if ('error' in result && result.error.name === 'ContextOverflowError')
          window.contextOverflows++
        return result
      }
    },
    { id: modelId, options: chatOptions(modelId) },
  )
  const model = new BrowserChatWebLLM(page, {
    modelId,
    thinkingPrefilled: true,
    profile: { maxInputTokens: 16384 },
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const agent = createDeepAgent({
    model,
    middleware: [
      createSummarizationMiddleware({
        model,
        backend: new StateBackend(),
        // Exercise native overflow before proactive summarization.
        trigger: { type: 'messages', value: 1000 },
        keep: { type: 'messages', value: 2 },
        trimTokensToSummarize: 512,
      }),
    ],
  })
  const result = await agent.invoke({
    messages: [
      new HumanMessage('These are old notes.\n' + 'field '.repeat(18000)),
      new AIMessage('Understood.'),
      new HumanMessage('Keep the old notes.'),
      new AIMessage('Ready.'),
      new HumanMessage('What is 2 + 2? Answer briefly.'),
    ],
  })
  expect(await page.evaluate(() => window.contextOverflows)).toBeGreaterThan(0)
  expect(Object.keys(result.files ?? {})).toContainEqual(
    expect.stringMatching(/^\/conversation_history\//),
  )
  expect(result.messages.at(-1)?.text).toContain('4')
  expect(result.messages.at(-1)?.content).toContainEqual(
    expect.objectContaining({ type: 'reasoning', reasoning: expect.any(String) }),
  )
})

declare global {
  interface Window {
    contextOverflows: number
  }
}
