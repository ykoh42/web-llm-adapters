import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../../shared/sampling.js'

test('browser transport stays close to native WebLLM on the same workload', async ({
  page,
}, testInfo) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
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
  const raw: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => raw.push(chunk),
  })
  const prompt = 'What is 2 + 2? Answer briefly.'
  await model.invoke(prompt)
  const request = await page.evaluate(
    () =>
      (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
        .nativeRequests[0]!,
  )
  const results: { native: number; adapter: number }[] = []
  for (let index = 0; index < 3; index++) {
    const native = await page.evaluate(async (request) => {
      const engine = window.checks.engine()
      await engine.resetChat()
      const start = performance.now()
      let text = ''
      const stream = await engine.chat.completions.create(request)
      for await (const chunk of stream) text += chunk.choices[0]?.delta.content ?? ''
      return { elapsed: performance.now() - start, text }
    }, request)
    await page.evaluate(() => window.checks.engine().resetChat())
    raw.length = 0
    const start = performance.now()
    await model.invoke(prompt)
    const elapsed = performance.now() - start
    const adapted = raw
      .map(
        (chunk) =>
          (chunk as import('@mlc-ai/web-llm').ChatCompletionChunk).choices[0]?.delta.content ?? '',
      )
      .join('')
    expect(adapted).toBe(native.text)
    results.push({ native: native.elapsed, adapter: elapsed })
  }
  const median = (values: number[]) => values.sort((a, b) => a - b)[1]!
  const baseline = median(results.map((result) => result.native))
  const adapted = median(results.map((result) => result.adapter))
  const path = testInfo.outputPath('performance.json')
  await writeFile(path, JSON.stringify({ modelId, results, ratio: adapted / baseline }, null, 2))
  await testInfo.attach('performance', { path, contentType: 'application/json' })
  expect(adapted).toBeLessThanOrEqual(baseline * 1.1 + 50)
})
