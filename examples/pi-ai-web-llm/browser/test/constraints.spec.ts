import { qwenTurns } from '../../../../test/support/qwen-output.js'
import { test, expect } from './browser.js'
import type {} from './checks.js'
import { modelCheck } from '../../../../scripts/release/support.js'

for (const modelId of [
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
]) {
  const thinking = !modelId.startsWith('Qwen2.5-')
  test(
    `${modelId} constrains an Agent tool loop and preserves native output`,
    modelCheck(modelId),
    async ({ page }, info) => {
      await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
      await page.evaluate((id) => window.piChecks.load(id), modelId)
      const results = await page.evaluate(() => window.piChecks.constrainedAgent())
      await info.attach('constrained-agent', {
        body: JSON.stringify(results),
        contentType: 'application/json',
      })
      for (const result of results) {
        expect(result.error).toBeUndefined()
        expect(result.inputs).toEqual([{ code: 'A7', enabled: false }])
        const assistants = result.messages.filter((message) => message.role === 'assistant')
        const calls = assistants.flatMap((message) =>
          message.content.filter((block) => block.type === 'toolCall'),
        )
        expect(calls).toMatchObject([{ name: 'lookup', arguments: { code: 'A7' } }])
        expect(calls).toHaveLength(1)
        const final = assistants
          .at(-1)
          ?.content.filter((block) => block.type === 'text')
          .map((block) => block.text)
          .join('')
        expect(final?.trim()).not.toBe('')
        const reasoning = assistants
          .flatMap((message) =>
            message.content
              .filter((block) => block.type === 'thinking')
              .map((block) => block.thinking),
          )
          .join('')
        const raw = qwenTurns(result.chunks, thinking && modelId.startsWith('Qwen3.5-')).join('')
        const text = assistants
          .flatMap((message) =>
            message.content.filter((block) => block.type === 'text').map((block) => block.text),
          )
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
