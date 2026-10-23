import { InMemoryCache } from '@langchain/core/caches'
import { tool } from '@langchain/core/tools'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

test('cached Gemma calls follow the current tool definition without extra inference', async ({
  page,
}, info) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    window.toolCacheRequests = []
    const engine = window.checks.engine()
    for (const endpoint of [engine.chat.completions, engine.completions]) {
      endpoint.create = new Proxy(endpoint.create, {
        apply(target, self, args) {
          window.toolCacheRequests.push(structuredClone(args[0]))
          return Reflect.apply(target, self, args)
        },
      })
    }
  })
  let raw: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    cache: new InMemoryCache(),
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => raw.push(chunk),
  })
  const definitions = ['lookup', 'search'].map((name) =>
    tool(() => '', {
      name,
      description: 'Look up an access code.',
      schema: z.object({ code: z.string() }),
    }),
  )
  const prompt = 'Use the provided tool to look up code A7.'
  const responses = []
  const native: WebLLMChunk[][] = []
  for (const [index, definition] of definitions.entries()) {
    raw = []
    const response = await model.bindTools([definition]).invoke(prompt)
    expect(response.tool_calls).toMatchObject([{ name: definition.name, args: { code: 'A7' } }])
    expect(response.contentBlocks).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: expect.stringMatching(/\S/) }),
    )
    expect(raw.length).toBeGreaterThan(0)
    responses.push(response)
    native.push(raw)
    expect(await page.evaluate(() => window.toolCacheRequests.length)).toBe(index + 1)
  }
  for (const [index, definition] of definitions.entries()) {
    for (const input of [definition, convertToOpenAITool(definition)]) {
      raw = []
      const cached = await model.bindTools([input]).invoke(prompt)
      expect(cached.content).toEqual(responses[index]!.content)
      expect(cached.tool_calls).toEqual(responses[index]!.tool_calls)
      expect(cached.usage_metadata?.total_tokens).toBe(0)
      expect(raw).toEqual([])
    }
  }
  const requests = await page.evaluate(() => window.toolCacheRequests)
  expect(requests).toHaveLength(2)
  for (const request of requests) {
    expect(request).toMatchObject({ model: modelId, max_tokens: 4096, seed: 7 })
    expect('prompt' in request && request.prompt).toContain('<|think|>')
  }
  await info.attach('cached-native-tools', {
    body: JSON.stringify(
      { requests, responses: responses.map((response) => response.toJSON()), native },
      null,
      2,
    ),
    contentType: 'application/json',
  })
})

declare global {
  interface Window {
    toolCacheRequests: (ChatCompletionRequestStreaming | CompletionCreateParamsStreaming)[]
  }
}
