import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'

test('Hermes named tool choice agrees with direct WebLLM or rejects its violation', async ({
  page,
}, info) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const engine = window.checks.engine()
    Object.assign(window, { nativeRequests: [] })
    engine.chat.completions.create = new Proxy(engine.chat.completions.create, {
      apply(target, self, args) {
        ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
          structuredClone(args[0]),
        )
        return Reflect.apply(target, self, args)
      },
    })
  })
  const adapted: ChatCompletionChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { temperature: 0, max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => {
      if (chunk.object === 'chat.completion.chunk') adapted.push(chunk)
    },
  })
  const bound = model.bindTools(
    ['weather', 'time'].map((name) => ({
      type: 'function' as const,
      function: {
        name,
        description: name === 'weather' ? 'Get the weather.' : 'Get the time.',
        parameters: { type: 'object', properties: {} },
      },
    })),
    { tool_choice: 'weather' },
  )
  const result = await bound.invoke('Use time to get the time.').then(
    (message) => ({ message, error: undefined }),
    (error: unknown) => ({ message: undefined, error: String(error) }),
  )
  const request = await page.evaluate(
    () =>
      (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
        .nativeRequests[0]!,
  )
  expect(request.tool_choice).toEqual({ type: 'function', function: { name: 'weather' } })
  expect(request.messages).toEqual([{ role: 'user', content: 'Use time to get the time.' }])
  expect(request.tools?.map((tool) => tool.function.name)).toEqual(['weather', 'time'])
  const direct = await page.evaluate(async (request) => {
    const engine = window.checks.engine()
    await engine.resetChat()
    const chunks = []
    for await (const chunk of await engine.chat.completions.create(request)) chunks.push(chunk)
    return chunks
  }, request)
  const content = (chunks: ChatCompletionChunk[]) =>
    chunks.map((chunk) => chunk.choices[0]?.delta.content ?? '').join('')
  const names = (chunks: ChatCompletionChunk[]) =>
    chunks
      .flatMap((chunk) => chunk.choices[0]?.delta.tool_calls ?? [])
      .map((call) => call.function?.name)
  await writeFile(
    info.outputPath('native-choice.json'),
    JSON.stringify({ request, direct, adapted, result }, null, 2),
  )
  expect(content(adapted)).toBe(content(direct))
  expect(names(adapted)).toEqual(names(direct))
  expect(names(direct).length).toBeGreaterThan(0)
  if (names(direct).some((name) => name !== 'weather')) {
    expect(result.error).toContain('outside tool_choice')
    expect(result.message).toBeUndefined()
    info.annotations.push({
      type: 'native limitation',
      description: 'WebLLM ignored named tool_choice.',
    })
  } else {
    expect(result.error).toBeUndefined()
    expect(result.message?.tool_calls?.map((call) => call.name)).toEqual(names(direct))
  }
})
