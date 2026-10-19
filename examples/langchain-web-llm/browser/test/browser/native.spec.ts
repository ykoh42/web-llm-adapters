import { createDeepAgent } from 'deepagents'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../../shared/sampling.js'
import { writeFile } from 'node:fs/promises'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'

test('Qwen3.5 0.8B: incomplete default-agent tool output also occurs without the adapter', async ({
  page,
}, testInfo) => {
  const modelId = 'Qwen3.5-0.8B-q4f16_1-MLC'
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
  const chunks: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: { maxInputTokens: 32768 },
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => chunks.push(chunk),
  })
  const agent = createDeepAgent({ model })
  await expect(
    agent.invoke({
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
    }),
  ).rejects.toThrow('inside a tool call')
  const adapted = chunks
    .map((chunk) =>
      'delta' in (chunk.choices[0] ?? {})
        ? ((chunk as import('@mlc-ai/web-llm').ChatCompletionChunk).choices[0]?.delta.content ?? '')
        : '',
    )
    .join('')
  const native = await page.evaluate(async () => {
    const engine = window.checks.engine()
    const request = (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
      .nativeRequests[0]!
    await engine.resetChat()
    const chunks = await engine.chat.completions.create(request)
    let text = ''
    for await (const chunk of chunks) text += chunk.choices[0]?.delta.content ?? ''
    return text
  })
  const path = testInfo.outputPath('comparison.json')
  await writeFile(path, JSON.stringify({ adapted, native }, null, 2))
  await testInfo.attach('native comparison', { path, contentType: 'application/json' })
  expect(native).toBe(adapted)
  expect(native).toContain('<tool_call>')
  expect(native).not.toContain('</tool_call>')
})
