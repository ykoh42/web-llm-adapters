import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { breakToolStream, expectedOutput } from '../../../../../test/support/stream-failure.js'

test('browser preserves a provider error status and can use the engine again', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const state = window as typeof window & { errorAttempts: number }
    state.errorAttempts = 0
    const engine = window.checks.engine()
    for (const endpoint of [engine.chat.completions, engine.completions]) {
      endpoint.create = new Proxy(endpoint.create, {
        apply(target, self, args) {
          if (++state.errorAttempts === 1)
            return Promise.reject({ message: 'Invalid request', status: 400 })
          return Reflect.apply(target, self, args)
        },
      })
    }
  })
  const model = new BrowserChatWebLLM(page, {
    modelId,
    maxRetries: 1,
    request: { ...sampling(modelId, false), max_tokens: 128 },
  })
  await expect(model.invoke('Hello')).rejects.toMatchObject({
    message: 'Invalid request',
    status: 400,
  })
  expect(
    await page.evaluate(() => (window as typeof window & { errorAttempts: number }).errorAttempts),
  ).toBe(1)
  expect((await model.invoke('What is 3 + 3? Reply briefly.')).text).toContain('6')
})

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
]) {
  test(`${modelId} preserves thinking and tool bytes through a stream failure`, async ({
    page,
  }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const gemma = modelId.startsWith('gemma-')
    await page.evaluate(breakToolStream, {
      app: 'langchain' as const,
      call: gemma ? '<|tool_call>' : '<tool_call>',
      after: gemma ? '<channel|>' : '</think>',
    })
    let native = ''
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => {
        const choice = chunk.choices[0]
        if (choice) native += 'delta' in choice ? (choice.delta.content ?? '') : choice.text
      },
    })
    const bound = model.bindTools([
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: 'Look up a code.',
          parameters: {
            type: 'object',
            properties: { code: { type: 'string' } },
            required: ['code'],
          },
        },
      },
    ])
    let text = ''
    let reasoning = ''
    await expect(
      (async () => {
        for await (const chunk of await bound.stream('Use lookup to look up code A7.')) {
          expect(chunk.tool_calls ?? []).toEqual([])
          for (const block of chunk.contentBlocks) {
            if (block.type === 'text') text += block.text
            if (block.type === 'reasoning') reasoning += block.reasoning
          }
        }
      })(),
    ).rejects.toThrow('Injected native stream failure')
    expect({ text, reasoning }).toEqual(
      expectedOutput(native, modelId, modelId.startsWith('Qwen3.5-')),
    )
    expect(reasoning.length).toBeGreaterThan(0)
    const recovered = await bound.invoke('Use lookup to look up code A7.')
    expect(recovered.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expect(recovered.contentBlocks.some((block) => block.type === 'reasoning')).toBe(true)
  })
}
