import { qwenTurns } from '../../../../../test/support/qwen-output.js'
import { createAgent, tool } from 'langchain'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { modelCheck } from '../../../../../scripts/release/support.js'

for (const modelId of [
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
]) {
  const thinking = !modelId.startsWith('Qwen2.5-')
  test(
    `${modelId} constrains an Agent tool loop and preserves native output`,
    thinking ? modelCheck(modelId, 'tools', 'thinking') : modelCheck(modelId, 'tools'),
    async ({ page }, info) => {
      await page.evaluate((id) => window.checks.load(id), modelId)
      for (const constrainedToolCalls of [false, true]) {
        const receipt = 'CODE-' + crypto.randomUUID().slice(0, 8)
        const inputs: unknown[] = []
        const chunks: WebLLMChunk[] = []
        const model = new BrowserChatWebLLM(page, {
          modelId,
          thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
          request: {
            ...sampling(modelId, thinking),
            constrainedToolCalls,
            max_tokens: 4096,
            seed: 7,
          },
          onChunk: (chunk) => chunks.push(chunk),
        })
        const lookup = tool(
          (input) => {
            inputs.push(input)
            return receipt
          },
          {
            name: 'lookup',
            description: 'Look up a code.',
            schema: z.strictObject({ code: z.string(), enabled: z.boolean() }),
          },
        )
        const result = await createAgent({ model, tools: [lookup] }).invoke(
          {
            messages: [
              {
                role: 'user',
                content:
                  'Call lookup with code A7 and enabled false. After its result, reply with the returned code.',
              },
            ],
          },
          { recursionLimit: 8 },
        )
        await info.attach('agent-' + constrainedToolCalls, {
          body: JSON.stringify({
            constrainedToolCalls,
            receipt,
            inputs,
            messages: result.messages,
            chunks,
          }),
          contentType: 'application/json',
        })
        expect(inputs).toEqual([{ code: 'A7', enabled: false }])
        expect(result.messages.at(-1)?.text.trim()).not.toBe('')
        const reasoning = result.messages
          .flatMap((message) =>
            Array.isArray(message.content)
              ? message.content
                  .filter((block) => block.type === 'reasoning')
                  .map((block) => block.reasoning)
              : [],
          )
          .join('')
        const raw = qwenTurns(chunks, thinking && modelId.startsWith('Qwen3.5-')).join('')
        const text = result.messages
          .filter((message) => message.type === 'ai')
          .map((message) => message.text)
          .join('')
        expect(text).toBe(
          thinking
            ? raw.replace(/<think>[\s\S]*?<\/think>|<tool_call>[\s\S]*?<\/tool_call>/g, '')
            : raw.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, ''),
        )
        expect(reasoning).toBe(
          thinking
            ? [...raw.matchAll(/<think>([\s\S]*?)<\/think>/g)].map((match) => match[1]).join('')
            : '',
        )
      }
    },
  )
}
