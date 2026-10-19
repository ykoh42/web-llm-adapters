import { sampling } from '../../../../shared/sampling.js'
import { createDeepAgent } from 'deepagents'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import type {} from '../../src/main.js'
import { writeFile } from 'node:fs/promises'

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
]) {
  test(`${modelId}: default Deep Agents tools and final answer`, async ({ page }, testInfo) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(() => {
      const engine = window.checks.engine()
      const create = engine.chat.completions.create.bind(engine.chat.completions)
      Object.assign(window, { nativeRequests: [] })
      engine.chat.completions.create = new Proxy(create, {
        apply(target, self, args) {
          ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
            structuredClone(args[0]),
          )
          return Reflect.apply(target, self, args)
        },
      })
    })
    const thinking = !modelId.startsWith('Qwen2.5-')
    const native: WebLLMChunk[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      profile: { maxInputTokens: 32768 },
      onChunk: (chunk) => native.push(chunk),
      request: { ...sampling(modelId, thinking), max_tokens: 4096, seed: 7 },
    })
    const agent = createDeepAgent({ model })
    let output: Awaited<ReturnType<typeof agent.invoke>>
    try {
      output = await agent.invoke({
        messages: [
          { role: 'user', content: 'Read /answer.txt and tell me the secret code exactly.' },
        ],
        files: {
          '/answer.txt': {
            content: ['The secret code is ORCHID-42.'],
            created_at: '2026-10-19',
            modified_at: '2026-10-19',
          },
        },
      })
    } finally {
      await writeFile(
        testInfo.outputPath('requests.json'),
        JSON.stringify(
          await page.evaluate(
            () => (window as unknown as { nativeRequests: unknown[] }).nativeRequests,
          ),
          null,
          2,
        ),
      )
      const path = testInfo.outputPath('native.json')
      await writeFile(path, JSON.stringify(native, null, 2))
      await testInfo.attach('native', {
        path,
        contentType: 'application/json',
      })
    }
    const result = output.messages.map((message) => ({
      type: message.type,
      text: message.text,
      content: message.content,
      ...('tool_calls' in message && { tool_calls: message.tool_calls }),
    }))
    await testInfo.attach('messages', {
      body: JSON.stringify(result, null, 2),
      contentType: 'application/json',
    })
    expect(
      result.some((message) => message.type === 'tool' && message.text.includes('ORCHID-42')),
    ).toBe(true)
    expect(result.at(-1)?.type).toBe('ai')
    expect(result.at(-1)?.text).toContain('ORCHID-42')
    if (!modelId.startsWith('Qwen2.5-'))
      expect(
        result.some(
          (message) =>
            Array.isArray(message.content) &&
            message.content.some(
              (block) =>
                block.type === 'reasoning' &&
                typeof block.reasoning === 'string' &&
                block.reasoning.length > 0,
            ),
        ),
      ).toBe(true)
  })
}
