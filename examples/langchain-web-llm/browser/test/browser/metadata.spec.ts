import { AIMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { createAgent } from 'langchain'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import type { ChatCompletionTokenLogprob } from '@mlc-ai/web-llm'
import { writeFile } from 'node:fs/promises'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

const cases = ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC'].flatMap((modelId) =>
  [false, true].map((logprobs) => ({ modelId, logprobs })),
)

for (const { modelId, logprobs } of cases) {
  test(`${modelId}: native metadata survives a tool loop (logprobs: ${logprobs})`, async ({
    page,
  }, info) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const native = new Map<string, { model: string; content: ChatCompletionTokenLogprob[] }>()
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: {
        ...sampling(modelId, true),
        max_tokens: 4096,
        seed: 7,
        ...(logprobs && { logprobs: true, top_logprobs: 2 }),
      },
      onChunk: (chunk) => {
        const response = native.get(chunk.id) ?? { model: chunk.model, content: [] }
        for (const token of chunk.choices[0]?.logprobs?.content ?? []) response.content.push(token)
        native.set(chunk.id, response)
      },
    })
    const lookup = tool(({ code }) => `The answer for ${code} is 42.`, {
      name: 'lookup',
      description: 'Look up the answer for a code.',
      schema: z.object({ code: z.string() }),
    })
    const agent = createAgent({ model, tools: [lookup] })
    const result = await agent
      .invoke({
        messages: [{ role: 'user', content: 'Use lookup to look up code A7. Tell me its answer.' }],
      })
      .finally(async () => {
        const path = info.outputPath('native.json')
        await writeFile(
          path,
          JSON.stringify([...native], (_key, value: unknown) =>
            typeof value === 'number' && !Number.isFinite(value) ? String(value) : value,
          ),
        )
        await info.attach('native', { path, contentType: 'application/json' })
      })

    expect(result.messages.map((message) => message.type)).toEqual(['human', 'ai', 'tool', 'ai'])
    const responses = result.messages.filter(AIMessage.isInstance)
    expect(native.size).toBe(2)
    expect(new Set(responses.map((message) => message.id)).size).toBe(2)
    for (const response of responses) {
      const original = native.get(response.id!)!
      expect(original.model).toBe(modelId)
      expect(response.response_metadata.model_name).toBe(modelId)
      if (logprobs) {
        expect(original.content.length).toBeGreaterThan(0)
        for (const token of original.content) expect(typeof token.logprob).toBe('number')
        expect(response.response_metadata.logprobs).toEqual({ content: original.content })
      } else expect(response.response_metadata).not.toHaveProperty('logprobs')
    }
    expect(responses[0]?.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expect(responses.at(-1)?.text).toContain('42')
    expect(
      responses.some((message) =>
        message.contentBlocks.some((block) => block.type === 'reasoning' && block.reasoning),
      ),
    ).toBe(true)
  })
}
