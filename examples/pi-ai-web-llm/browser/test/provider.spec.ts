import { writeFile } from 'node:fs/promises'
import type { AssistantMessage } from '@earendil-works/pi-ai'
import { test, expect } from './browser.js'
import type {} from './checks.js'
import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMStreamOptions } from 'pi-ai-web-llm'

test('Pi rejects invalid tool choices before inference and keeps auto and none usable', async ({
  page,
}, info) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const results = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const agent = await window.piChecks.createAgent(true)
    const context = {
      systemPrompt: agent.state.systemPrompt,
      tools: agent.state.tools,
      messages: [
        { role: 'user' as const, content: 'Use lookup to look up code A7.', timestamp: 0 },
      ],
    }
    let payloadCalls = 0
    const invalid = []
    for (const simple of [false, true]) {
      for (const toolChoice of ['required', 'lookup', 'none ', null]) {
        const options: WebLLMStreamOptions = {
          reasoning: 'low',
          onPayload: () => {
            payloadCalls++
          },
        }
        Object.assign(options, { toolChoice })
        invalid.push(
          simple
            ? await models.completeSimple(model, context, options)
            : await models.complete(model, context, options),
        )
      }
    }
    const blockedPayloads = payloadCalls
    const valid = []
    for (const toolChoice of ['auto', 'none'] as const) {
      valid.push(
        await models.completeSimple(model, context, {
          toolChoice,
          reasoning: 'low',
          onPayload: () => {
            payloadCalls++
          },
        }),
      )
    }
    return { invalid, valid, blockedPayloads, payloadCalls }
  })
  await info.attach('choices', {
    body: JSON.stringify(results, null, 2),
    contentType: 'application/json',
  })
  expect(results.blockedPayloads).toBe(0)
  expect(results.payloadCalls).toBe(2)
  for (const result of results.invalid)
    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: 'toolChoice must be auto or none.',
      content: [],
    })
  expect(results.valid[0]).toMatchObject({
    stopReason: 'toolUse',
    content: expect.arrayContaining([
      { type: 'toolCall', id: expect.any(String), name: 'lookup', arguments: { code: 'A7' } },
    ]),
  })
  expect(results.valid[1]?.stopReason).toBe('stop')
  expect(results.valid[1]?.content.some((part) => part.type === 'toolCall')).toBe(false)
})

test('Pi applies thinking-level sampling and cancels pending callbacks on a real Worker', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const results = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(false)
    const selected = {
      ...model,
      samplingParams: { temperature: 0.7 },
      samplingParamsByThinkingLevel: {
        off: { temperature: 0 },
        high: { temperature: 0.1 },
      },
    }
    const context = {
      messages: [{ role: 'user' as const, content: 'What is 3 + 3? Reply briefly.', timestamp: 0 }],
    }
    const payloads: unknown[] = []
    const responses = []
    for (const reasoning of [undefined, 'high'] as const)
      responses.push(
        await models.completeSimple(selected, context, {
          ...(reasoning && { reasoning }),
          onPayload: (payload) => {
            payloads.push(structuredClone(payload))
          },
        }),
      )
    const cancelled = []
    const recovered = []
    for (const callback of ['onPayload', 'onProviderStreamEvent'] as const) {
      for (const timeout of [false, true]) {
        const controller = new AbortController()
        let finish!: () => void
        const hold = new Promise<void>((resolve) => (finish = resolve))
        try {
          cancelled.push(
            await models.completeSimple(model, context, {
              signal: controller.signal,
              ...(timeout && { timeoutMs: 1000 }),
              [callback]: () => {
                if (!timeout) controller.abort()
                return hold
              },
            }),
          )
          recovered.push(await models.completeSimple(model, context))
        } finally {
          finish()
        }
      }
    }
    return { payloads, responses, cancelled, recovered }
  })
  expect(results.payloads).toMatchObject([{ temperature: 0 }, { temperature: 0.1 }])
  for (const message of [...results.responses, ...results.recovered]) {
    expect(message.stopReason).toBe('stop')
    expect(
      message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    ).toContain('6')
  }
  expect(results.cancelled.map((message) => message.stopReason)).toEqual(Array(4).fill('aborted'))
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'Qwen2.5-3B-Instruct-q4f16_1-MLC']) {
  test(`${modelId}: bare JSON mode preserves native output`, async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const result = await page.evaluate(async () => {
      const { models, model } = await window.piChecks.provider(false)
      let native = ''
      let payload: unknown
      const message = await models.complete(
        model,
        {
          messages: [
            { role: 'user', content: 'Return a JSON object with answer equal to 6.', timestamp: 0 },
          ],
        },
        {
          request: { response_format: { type: 'json_object' } },
          onPayload: (request) => {
            payload = structuredClone(request)
          },
          onProviderStreamEvent: (event) => {
            const chunk = event as ChatCompletionChunk
            native += chunk.choices[0]?.delta.content ?? ''
          },
        },
      )
      const text = message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('')
      return { message, native, text, payload }
    })
    expect(result.message.stopReason).toBe('stop')
    expect(result.payload).toMatchObject({ response_format: { type: 'json_object', schema: '{}' } })
    expect(result.native.length).toBeGreaterThan(0)
    expect(result.text).toBe(result.native)
    if (modelId.startsWith('Qwen2.5-')) expect(JSON.parse(result.text)).toEqual({ answer: 6 })
  })
}

test('the example UI loads a Worker and runs the Pi tool loop without test globals', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('http://127.0.0.1:4173/pi/')
  expect(await page.evaluate(() => typeof window.piChecks)).toBe('undefined')
  await page.getByRole('button', { name: 'Load model', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('Ready', { timeout: 15 * 60 * 1000 })
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('stop', { timeout: 15 * 60 * 1000 })
  await expect(page.locator('#answer')).toContainText('ORCHID-42')
  await expect(page.locator('#details')).toContainText('toolResult')
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled()
  expect(errors).toEqual([])
})

for (const modelId of process.env.MODEL_IDS?.split(',') ?? [
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
]) {
  test(`${modelId}: Pi Agent Core tools, usage, cancellation and recovery in a Worker`, async ({
    page,
  }, info) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const evidence = await page.evaluate(async () => {
      const agent = await window.piChecks.createAgent(true)
      const requests: unknown[] = []
      const native: ChatCompletionChunk[] = []
      agent.onPayload = (payload) => {
        requests.push(structuredClone(payload))
      }
      agent.onProviderStreamEvent = (event) => {
        native.push(structuredClone(event) as ChatCompletionChunk)
      }
      const error = await agent.prompt('Look up A7 and tell me the answer.').then(
        () => null,
        (error: unknown) => String(error),
      )
      return { messages: agent.state.messages, requests, native, error }
    })
    const path = info.outputPath('model.json')
    await writeFile(path, JSON.stringify(evidence, null, 2))
    await info.attach('messages', {
      path,
      contentType: 'application/json',
    })
    expect(evidence.error).toBeNull()
    const { messages } = evidence
    expect(
      messages.some(
        (message) =>
          message.role === 'assistant' &&
          message.content.some(
            (part) =>
              part.type === 'toolCall' && part.name === 'lookup' && part.arguments.code === 'A7',
          ),
      ),
    ).toBe(true)
    expect(
      messages.some(
        (message) =>
          message.role === 'toolResult' && message.toolName === 'lookup' && !message.isError,
      ),
    ).toBe(true)
    const final = messages.at(-1)
    expect(final).toMatchObject({ role: 'assistant', stopReason: 'stop' })
    if (final?.role === 'assistant') {
      expect(
        final.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join(''),
      ).toContain('ORCHID-42')
      expect(final.usage.totalTokens).toBeGreaterThan(0)
    }
    if (!modelId.startsWith('Qwen2.5-'))
      expect(
        messages.some(
          (message) =>
            message.role === 'assistant' &&
            message.content.some((part) => part.type === 'thinking' && part.thinking.length > 0),
        ),
      ).toBe(true)
    const recovery = await page.evaluate(() => window.piChecks.cancelAndRecover())
    await info.attach('cancellation', {
      body: JSON.stringify(recovery, null, 2),
      contentType: 'application/json',
    })
    expect(recovery.observed).toBe(true)
    expect(recovery.cancelled.stopReason).toBe('aborted')
    expect(recovery.recovered.message.stopReason).toBe('stop')
    expect(recovery.recovered.events).toContain('text_delta')
    expect(
      recovery.recovered.message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    ).toContain('6')
    expect(errors).toEqual([])
  })
}

for (const modelId of ['Qwen3.5-9B-q4f16_1-MLC', 'Qwen3.5-2B-q4f16_1-MLC']) {
  test(`${modelId}: thinking and answer match native WebLLM output`, async ({ page }, info) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const evidence = await page.evaluate(async () => {
      const { models, model } = await window.piChecks.provider(true)
      let raw = ''
      const message = await models.completeSimple(
        model,
        {
          messages: [{ role: 'user', content: 'What is 17 * 23 + 19 * 29?', timestamp: 0 }],
        },
        {
          reasoning: 'low',
          onProviderStreamEvent: (event) => {
            const chunk = event as ChatCompletionChunk
            raw += chunk.choices[0]?.delta.content ?? ''
          },
        },
      )
      return { message, raw }
    })
    await writeFile(info.outputPath('thinking.json'), JSON.stringify(evidence, null, 2))
    const { message, raw } = evidence
    const [, thinking, text] = /^<think>([\s\S]*?)<\/think>([\s\S]*)$/.exec(raw) ?? []
    expect(thinking).toBeDefined()
    expect(message.stopReason).toBe('stop')
    const answer = message.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('')
    expect(answer).toContain('942')
    expect(answer).toBe(text)
    expect(
      message.content
        .filter((part) => part.type === 'thinking')
        .map((part) => part.thinking)
        .join(''),
    ).toBe(thinking)
  })
}

for (const modelId of [
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
]) {
  test(`${modelId}: native thinking sampling overrides named Pi reasoning on a real Worker`, async ({
    page,
  }, info) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const results = await page.evaluate(async () => {
      const { models, model, engine } = await window.piChecks.provider(true)
      const context = {
        messages: [{ role: 'user' as const, content: 'What is 17 * 23?', timestamp: 0 }],
      }
      const results: {
        simple: boolean
        enable_thinking: boolean
        message: AssistantMessage
        payload: ChatCompletionRequestStreaming
        native: string
        direct: string
      }[] = []
      for (const simple of [false, true]) {
        for (const enable_thinking of [true, false]) {
          let payload: ChatCompletionRequestStreaming | undefined
          let native = ''
          const options: WebLLMStreamOptions = {
            ...(!enable_thinking && { reasoning: 'high' }),
            samplingParams: { extra_body: { enable_thinking, enable_latency_breakdown: true } },
            onPayload: (request) => {
              payload = structuredClone(request) as ChatCompletionRequestStreaming
            },
            onProviderStreamEvent: (event) => {
              const chunk = event as ChatCompletionChunk
              native += chunk.choices[0]?.delta.content ?? ''
            },
          }
          const message = simple
            ? await models.completeSimple(model, context, options)
            : await models.complete(model, context, options)
          if (!payload) throw new Error('Missing native request.')
          results.push({ simple, enable_thinking, message, payload, native, direct: '' })
        }
      }
      await window.piChecks.withEngine(engine, async (engine) => {
        await engine.resetChat()
        for (const result of results) {
          for await (const chunk of await engine.chat.completions.create(result.payload))
            result.direct += chunk.choices[0]?.delta.content ?? ''
        }
      })
      return results
    })
    await writeFile(info.outputPath('thinking-sampling.json'), JSON.stringify(results, null, 2))
    expect(results).toHaveLength(4)
    for (const { simple, enable_thinking, message, payload, native, direct } of results) {
      const mode = `simple=${simple}, thinking=${enable_thinking}`
      expect(payload, mode).toMatchObject({
        max_tokens: 4096,
        seed: 7,
        extra_body: { enable_thinking, enable_latency_breakdown: true },
      })
      expect(native, mode).toBe(direct)
      expect(message.stopReason, mode).toBe('stop')
      const text = message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('')
      expect(text, mode).toContain('391')
      expect(text, mode).toBe(direct.replace(/^<think>[\s\S]*?<\/think>/, ''))
      const thinking = message.content
        .filter((part) => part.type === 'thinking')
        .map((part) => part.thinking)
        .join('')
      const nativeThinking = /<think>([\s\S]*?)<\/think>/.exec(direct)?.[1] ?? ''
      expect(thinking, mode).toBe(nativeThinking)
      if (enable_thinking && modelId.startsWith('Qwen3-')) expect(thinking, mode).toMatch(/\S/)
      if (!enable_thinking) expect(thinking, mode).toMatch(/^\s*$/)
    }
  })
}
