import { AIMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { InMemoryStore } from '@langchain/langgraph'
import { createAgent } from 'langchain'
import { BrowserChatWebLLM, BrowserWebLLMEmbeddings } from 'langchain-web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

test('an agent searches real vector memory with chat and embeddings on one engine', async ({
  page,
}) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  const embeddingId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(
    async ({ modelId, embeddingId }) => {
      const engine = window.checks.engine()
      await engine.reload([modelId, embeddingId], [{ context_window_size: 16384 }, {}])
      window.memoryRequests = []
      for (const [endpoint, api] of [
        ['chat', engine.chat.completions],
        ['embedding', engine.embeddings],
      ] as const) {
        api.create = new Proxy(api.create, {
          apply(target, self, args) {
            window.memoryRequests.push({ endpoint, model: args[0].model })
            return Reflect.apply(target, self, args)
          },
        })
      }
    },
    { modelId, embeddingId },
  )
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: embeddingId })
  const store = new InMemoryStore({ index: { embeddings, dims: 384 } })
  await store.put(['notes'], 'orchid', { text: 'The access code for the orchid is ORCHID-42.' })
  await store.put(['notes'], 'car', { text: 'The access code for the car is CAR-17.' })
  await page.evaluate(() => {
    window.memoryRequests = []
  })
  const retrieved: string[] = []
  const lookup = tool(
    async ({ query }) => {
      const results = await store.search(['notes'], { query, limit: 1 })
      retrieved.push(...results.map((result) => result.key))
      return JSON.stringify(results.map((result) => result.value))
    },
    {
      name: 'lookup_memory',
      description: 'Search saved notes for an access code.',
      schema: z.object({ query: z.string() }),
    },
  )
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: { maxInputTokens: 16384 },
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const agent = createAgent({ model, tools: [lookup] })
  const result = await agent.invoke({
    messages: [
      { role: 'user', content: 'Use lookup_memory to find the orchid code. Reply with the code.' },
    ],
  })
  expect(retrieved).toEqual(['orchid'])
  expect(result.messages.at(-1)?.text).toContain('ORCHID-42')
  expect(result.messages.at(-1)?.response_metadata.finish_reason).toBe('stop')
  expect(
    result.messages.some(
      (message) =>
        AIMessage.isInstance(message) &&
        message.contentBlocks.some((block) => block.type === 'reasoning'),
    ),
  ).toBe(true)
  expect(await page.evaluate(() => window.memoryRequests)).toEqual([
    { endpoint: 'chat', model: modelId },
    { endpoint: 'embedding', model: embeddingId },
    { endpoint: 'chat', model: modelId },
  ])
})

declare global {
  interface Window {
    memoryRequests: { endpoint: 'chat' | 'embedding'; model: string | null | undefined }[]
  }
}
