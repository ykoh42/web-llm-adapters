import { test, expect } from './browser.js'
import type {} from './checks.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'

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
  expect(result.native).toMatch(/^<think>/)
  expect(result.native).not.toContain('</think>')
  expect(reasoning).not.toBe('')
  expect(reasoning).toBe(result.native.slice('<think>'.length))
  expect(result.response.content.every((block) => block.type === 'thinking')).toBe(true)
})

test('unset sampling token limits match native payload handling with thinking enabled', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const context = {
      messages: [{ role: 'user' as const, content: 'What is 3 + 3? Reply briefly.', timestamp: 0 }],
    }
    const baseline = await models.complete(model, context, {
      reasoning: 'low',
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, max_tokens: undefined }
      },
    })
    let unset = false
    const response = await models.complete(model, context, {
      reasoning: 'low',
      samplingParams: { max_tokens: undefined },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        unset = 'max_tokens' in payload && payload.max_tokens === undefined
      },
    })
    return { baseline, response, unset }
  })

  expect(result.unset).toBe(true)
  expect(result.baseline.stopReason).toBe('stop')
  expect(result.response.stopReason).toBe('stop')
  expect(result.response.content).toEqual(result.baseline.content)
  expect(result.response.usage).toEqual(result.baseline.usage)
  expect(result.response.content.some((part) => part.type === 'thinking' && part.thinking)).toBe(
    true,
  )
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
    const { models, model } = await window.piChecks.provider(true)
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
    const baseline = await models.complete(model, reasoningContext, {
      reasoning: 'low',
      request: { response_format: text },
    })
    const changed = await models.complete(model, reasoningContext, {
      reasoning: 'low',
      request: { response_format: json },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, response_format: text }
      },
    })
    return { structured, native, baseline, changed }
  })

  expect(result.structured.stopReason).toBe('stop')
  expect(result.native.length).toBeGreaterThan(0)
  expect(
    result.structured.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join(''),
  ).toBe(result.native)
  expect(result.baseline.stopReason).toBe('stop')
  expect(result.changed.stopReason).toBe('stop')
  expect(result.changed.content).toEqual(result.baseline.content)
  expect(result.changed.content.some((part) => part.type === 'thinking' && part.thinking)).toBe(
    true,
  )
})

test('Gemma honors a stop string selected by onPayload inside thinking', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('gemma-4-E2B-it-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const context = {
      messages: [
        {
          role: 'user' as const,
          content: 'What is 17 * 23? Explain your calculation.',
          timestamp: 0,
        },
      ],
    }
    const stop = ['<channel|>']
    const baseline = await models.complete(model, context, { reasoning: 'low', request: { stop } })
    const changed = await models.complete(model, context, {
      reasoning: 'low',
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, stop }
      },
    })
    return { baseline, changed }
  })

  expect(result.baseline.stopReason).toBe('stop')
  expect(result.changed.stopReason).toBe('stop')
  expect(result.changed.content).toEqual(result.baseline.content)
  expect(result.changed.content.some((part) => part.type === 'thinking' && part.thinking)).toBe(
    true,
  )
  expect(result.changed.content.every((part) => part.type === 'thinking')).toBe(true)
})
