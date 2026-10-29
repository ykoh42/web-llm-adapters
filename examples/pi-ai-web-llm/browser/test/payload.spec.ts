import { test, expect } from './browser.js'
import type {} from './checks.js'
import type {
  ChatCompletionChunk,
  ChatCompletionRequestStreaming,
  Completion,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { nativeOutput } from '../../../../test/support/native-output.js'

test('Qwen preserves a caller stop inside thinking selected by onPayload', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const context = {
      messages: [{ role: 'user' as const, content: 'What is 17 * 23?', timestamp: 0 }],
    }
    let native = ''
    const finishes: string[] = []
    const response = await models.complete(model, context, {
      reasoning: 'low',
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, stop: ['</think>'] }
      },
      onProviderStreamEvent: (event) => {
        const chunk = event as ChatCompletionChunk
        native += chunk.choices[0]?.delta.content ?? ''
        const finish = chunk.choices[0]?.finish_reason
        if (finish) finishes.push(finish)
      },
    })
    return { response, native, finishes }
  })
  const reasoning = result.response.content
    .filter((block) => block.type === 'thinking')
    .map((block) => block.thinking)
    .join('')

  expect(result.response.stopReason).toBe('stop')
  expect(result.finishes).toEqual(['stop'])
  expect(result.native).not.toMatch(/^<think>/)
  expect(result.native).not.toContain('</think>')
  expect(reasoning).not.toBe('')
  expect(reasoning).toBe(result.native)
  expect(result.response.content.every((block) => block.type === 'thinking')).toBe(true)
})

test('unset sampling token limits reach WebLLM with thinking enabled', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate((id) => window.piChecks.load(id), modelId)
  const result = await page.evaluate(async () => {
    const { models, model, engine } = await window.piChecks.provider(true)
    let request: ChatCompletionRequestStreaming | undefined
    const endpoint = engine.chat.completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        request = structuredClone(args[0])
        return Reflect.apply(target, self, args)
      },
    })
    const chunks: ChatCompletionChunk[] = []
    const response = await models.complete(
      model,
      { messages: [{ role: 'user', content: 'What is 3 + 3? Reply briefly.', timestamp: 0 }] },
      {
        reasoning: 'low',
        samplingParams: { max_tokens: undefined },
        onProviderStreamEvent: (event) => {
          chunks.push(event as ChatCompletionChunk)
        },
      },
    )
    return { response, request, chunks }
  })

  expect(result.request).toHaveProperty('max_tokens', undefined)
  expect(result.request?.extra_body?.enable_thinking).toBe(true)
  expect(result.response.stopReason).toBe('stop')
  const text = result.response.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('')
  const thinking = result.response.content
    .filter((part) => part.type === 'thinking')
    .map((part) => part.thinking)
    .join('')
  expect(thinking).not.toBe('')
  expect({ text, thinking: [thinking] }).toEqual(nativeOutput(result.chunks, modelId, '<think>'))
  const usage = result.chunks.at(-1)?.usage
  expect(usage).toBeDefined()
  expect(result.response.usage).toMatchObject({
    input: usage?.prompt_tokens,
    output: usage?.completion_tokens,
    totalTokens: usage?.total_tokens,
  })
})

test('unreadable callback errors finish and leave a real WebLLM engine usable', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const responses = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const context = {
      messages: [{ role: 'user' as const, content: 'What is 3 + 3? Reply briefly.', timestamp: 0 }],
    }
    const responses = []
    for (const callback of ['onPayload', 'onProviderStreamEvent'] as const) {
      const failed = await models.complete(model, context, {
        [callback]: () => {
          throw Object.create(null)
        },
      })
      const recovered = await models.complete(model, context)
      responses.push({ callback, failed, recovered })
    }
    return responses
  })

  for (const { failed, recovered } of responses) {
    expect(failed).toMatchObject({
      stopReason: 'error',
      errorMessage: 'The request failed with an unreadable error.',
      content: [],
    })
    expect(recovered.stopReason).toBe('stop')
    expect(
      recovered.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    ).toContain('6')
  }
})

test('Qwen uses the response format selected by onPayload without losing native bytes', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model, engine } = await window.piChecks.provider(true)
    const requests: ChatCompletionRequestStreaming[] = []
    const endpoint = engine.chat.completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        requests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
    const context = {
      messages: [
        {
          role: 'user' as const,
          content: 'Return JSON with text equal to the string "<think>literal</think>".',
          timestamp: 0,
        },
      ],
    }
    const json = { type: 'json_object' as const, schema: '{}' }
    const text = { type: 'text' as const }
    let native = ''
    const structured = await models.complete(model, context, {
      request: { response_format: text },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, response_format: json }
      },
      onProviderStreamEvent: (event) => {
        const chunk = event as ChatCompletionChunk
        native += chunk.choices[0]?.delta.content ?? ''
      },
    })
    const reasoningContext = {
      messages: [{ role: 'user' as const, content: 'What is 17 * 23?', timestamp: 0 }],
    }
    const chunks: ChatCompletionChunk[] = []
    const changed = await models.complete(model, reasoningContext, {
      reasoning: 'low',
      request: { response_format: json },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, response_format: text }
      },
      onProviderStreamEvent: (event) => {
        chunks.push(event as ChatCompletionChunk)
      },
    })
    return { structured, native, changed, chunks, requests }
  })

  expect(result.requests.map((request) => request.response_format)).toEqual([
    { type: 'json_object', schema: '{}' },
    { type: 'text' },
  ])
  expect(result.structured.stopReason).toBe('stop')
  expect(result.native.length).toBeGreaterThan(0)
  expect(
    result.structured.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join(''),
  ).toBe(result.native)
  expect(result.changed.stopReason).toBe('stop')
  const text = result.changed.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('')
  const thinking = result.changed.content
    .filter((part) => part.type === 'thinking')
    .map((part) => part.thinking)
    .join('')
  expect(thinking).not.toBe('')
  expect({ text, thinking: [thinking] }).toEqual(
    nativeOutput(result.chunks, 'Qwen3.5-4B-q4f16_1-MLC', '<think>'),
  )
})

test('Gemma honors a stop string selected by onPayload inside thinking', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('gemma-4-E2B-it-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model, engine } = await window.piChecks.provider(true)
    let request: CompletionCreateParamsStreaming | undefined
    const endpoint = engine.completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        request = structuredClone(args[0])
        return Reflect.apply(target, self, args)
      },
    })
    const chunks: Completion[] = []
    const response = await models.complete(
      model,
      {
        messages: [
          {
            role: 'user',
            content: 'What is 17 * 23? Explain your calculation.',
            timestamp: 0,
          },
        ],
      },
      {
        reasoning: 'low',
        onPayload: (payload) => {
          if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
          return { ...payload, stop: ['<channel|>'] }
        },
        onProviderStreamEvent: (event) => {
          chunks.push(event as Completion)
        },
      },
    )
    return { response, request, chunks }
  })

  expect(result.request?.stop).toEqual(['<channel|>'])
  expect(result.response.stopReason).toBe('stop')
  expect(result.response.content.every((part) => part.type === 'thinking')).toBe(true)
  const thinking = result.response.content
    .filter((part) => part.type === 'thinking')
    .map((part) => part.thinking)
    .join('')
  const native = result.chunks
    .flatMap((chunk) => chunk.choices.map((choice) => choice.text))
    .join('')
  expect(thinking).not.toBe('')
  expect(native).not.toContain('<channel|>')
  const prefix = '<|channel>thought\n'
  if (!native.startsWith(prefix)) expect(result.request?.prompt).toMatch(/<\|channel>thought\n$/)
  expect(thinking).toBe(native.replace(/^<\|channel>thought\n/, ''))
})
