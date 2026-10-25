import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { InMemoryCache } from '@langchain/core/caches'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming

test('Qwen preserves a caller stop inside thinking', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const chunks: WebLLMChunk[] = []
  const request = {
    ...sampling(modelId, true),
    max_tokens: 4096,
    seed: 7,
    stop: ['</think>'],
  }
  const model = new BrowserChatWebLLM(page, {
    modelId,
    thinkingPrefilled: true,
    request,
    onChunk: (chunk) => chunks.push(chunk),
  })
  const response = await model.invoke('What is 17 * 23?')
  const native = summarize(chunks)
  const reasoning = response.contentBlocks
    .filter((block) => block.type === 'reasoning')
    .map((block) => block.reasoning)
    .join('')

  expect(native.finishes).toEqual(['stop'])
  expect(native.text).not.toMatch(/^<think>/)
  expect(native.text).not.toContain('</think>')
  expect(reasoning).not.toBe('')
  expect(reasoning).toBe(native.text)
  expect(response.text).toBe('')
  expect(response.response_metadata.finish_reason).toBe('stop')
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId}: sampling options and log probabilities match direct WebLLM`, async ({
    page,
  }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(() => {
      const engine = window.checks.engine()
      Object.assign(window, { samplingRequests: [] })
      for (const endpoint of [engine.chat.completions, engine.completions]) {
        endpoint.create = new Proxy(endpoint.create, {
          apply(target, self, args) {
            window.samplingRequests.push(structuredClone(args[0]))
            return Reflect.apply(target, self, args)
          },
        })
      }
    })
    const chunks: WebLLMChunk[] = []
    const request = {
      ...sampling(modelId, true),
      max_tokens: 4096,
      seed: 7,
      frequency_penalty: 0.1,
      presence_penalty: 0.1,
      logit_bias: { '0': -0.25 },
      logprobs: true,
      top_logprobs: 2,
      extra_body: { enable_thinking: true, enable_latency_breakdown: true },
    }
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request,
      onChunk: (chunk) => chunks.push(chunk),
    })
    const result = await model.invoke('What is 17 * 23? Reply briefly.')
    const prepared = await page.evaluate(() => window.samplingRequests[0]!)
    for (const key of [
      'frequency_penalty',
      'presence_penalty',
      'logit_bias',
      'logprobs',
      'top_logprobs',
    ] as const)
      expect(prepared[key]).toEqual(request[key])
    expect(prepared.extra_body?.enable_latency_breakdown).toBe(true)
    const native = await page.evaluate(async (input) => {
      const engine = window.checks.engine()
      await engine.resetChat()
      const stream =
        'messages' in input
          ? await engine.chat.completions.create(input)
          : await engine.completions.create(input)
      const chunks: WebLLMChunk[] = []
      for await (const chunk of stream) chunks.push(chunk)
      return chunks
    }, prepared)
    const observed = summarize(chunks)
    const expected = summarize(native)
    expect({ text: observed.text, finishes: observed.finishes, tokens: observed.tokens }).toEqual({
      text: expected.text,
      finishes: expected.finishes,
      tokens: expected.tokens,
    })
    expectLogprobs(observed.probabilities, expected.probabilities)
    expect(observed.finishes).toEqual(['stop'])
    expect(observed.probabilities.length).toBeGreaterThan(0)
    expect(observed.probabilities.every((item) => item.top_logprobs.length === 2)).toBe(true)
    expect(result.text).toContain('391')
    expect(result.contentBlocks.some((block) => block.type === 'reasoning')).toBe(true)
    expect(result.usage_metadata?.output_tokens).toBe(observed.tokens)
  })
}

test('cached native Hermes tool calls keep constructor and call output formats', async ({
  page,
}) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const endpoint = window.checks.engine().chat.completions
    Object.assign(window, { samplingRequests: [] })
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.samplingRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
  const cache = new InMemoryCache()
  const models = (['v0', 'v1'] as const).map((outputVersion) =>
    new BrowserChatWebLLM(page, {
      modelId,
      cache,
      outputVersion,
      request: { max_tokens: 4096, temperature: 0, seed: 7 },
    }).bindTools(
      [
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
      ],
      { tool_choice: 'lookup' },
    ),
  )
  const input = 'Call lookup for code A7.'
  const expected = []
  for (const model of models) {
    const answer = await model.invoke(input)
    expect(answer.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expected.push(structuredClone({ content: answer.content, calls: answer.tool_calls }))
  }
  expect(expected[0]!.content).not.toEqual(expected[1]!.content)
  for (const [index, model] of models.entries()) {
    const answer = await model.invoke(input)
    expect({ content: answer.content, calls: answer.tool_calls }).toEqual(expected[index])
    for (const outputVersion of [undefined, null]) {
      const options = {}
      Object.assign(options, { outputVersion })
      const inherited = await model.invoke(input, options)
      expect({ content: inherited.content, calls: inherited.tool_calls }).toEqual(expected[index])
    }
    const other = index === 0 ? 'v1' : 'v0'
    const selected = await model.invoke(input, { outputVersion: other })
    expect({ content: selected.content, calls: selected.tool_calls }).toEqual(expected[1 - index])
  }
  const requests = await page.evaluate(() => window.samplingRequests)
  expect(requests).toHaveLength(2)
  expect(requests[1]).toEqual(requests[0])
  const native = await page.evaluate(
    (id) =>
      window.webllmLangChainBridge!.withEngine(async (engine) => ({
        message: await engine.getMessage(id),
        stats: await engine.runtimeStatsText(id),
      })),
    modelId,
  )
  expect(native.message).toContain('A7')
  expect(native.stats.length).toBeGreaterThan(0)
})

function summarize(chunks: WebLLMChunk[]) {
  const choices = chunks.flatMap<WebLLMChunk['choices'][number]>((chunk) => chunk.choices)
  return {
    text: choices
      .map((choice) => ('delta' in choice ? (choice.delta.content ?? '') : choice.text))
      .join(''),
    probabilities: choices.flatMap((choice) => choice.logprobs?.content ?? []),
    finishes: choices.flatMap((choice) => (choice.finish_reason ? [choice.finish_reason] : [])),
    tokens: chunks.filter((chunk) => chunk.usage).at(-1)?.usage?.completion_tokens,
  }
}

function expectLogprobs(
  actual: ReturnType<typeof summarize>['probabilities'],
  expected: ReturnType<typeof summarize>['probabilities'],
) {
  expect(actual).toHaveLength(expected.length)
  for (const [index, item] of actual.entries()) {
    const other = expected[index]!
    expect({ token: item.token, bytes: item.bytes }).toEqual({
      token: other.token,
      bytes: other.bytes,
    })
    // GPU reductions can shift probabilities slightly between identical runs.
    expect(item.logprob).toBeCloseTo(other.logprob, 1)
    expect(item.top_logprobs).toHaveLength(other.top_logprobs.length)
    for (const [topIndex, top] of item.top_logprobs.entries()) {
      const expectedTop = other.top_logprobs[topIndex]!
      expect({ token: top.token, bytes: top.bytes }).toEqual({
        token: expectedTop.token,
        bytes: expectedTop.bytes,
      })
      expect(top.logprob).toBeCloseTo(expectedTop.logprob, 1)
    }
  }
}

declare global {
  interface Window {
    samplingRequests: NativeRequest[]
  }
}
