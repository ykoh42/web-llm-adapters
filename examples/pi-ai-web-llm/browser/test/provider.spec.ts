import { modelCheck } from '../../../../scripts/release/support.js'
import { writeFile } from 'node:fs/promises'
import type { AssistantMessage, SimpleStreamOptions } from '@earendil-works/pi-ai'
import { test, expect } from './browser.js'
import type {} from './checks.js'
import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMChunk } from 'pi-ai-web-llm'
import { nativeOutput } from '../../../../test/support/native-output.js'

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
        const options: SimpleStreamOptions = {
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
      errorMessage: 'toolChoice must be auto, none, or a function choice.',
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

test('Pi API calls can select one registered function on Qwen3.5', async ({ page }, info) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-2B-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const agent = await window.piChecks.createAgent(true)
    let request: unknown
    const message = await models.complete(
      model,
      {
        systemPrompt: agent.state.systemPrompt,
        tools: agent.state.tools,
        messages: [
          { role: 'user', content: 'Use lookup to look up code A7.', timestamp: Date.now() },
        ],
      },
      {
        toolChoice: { type: 'function', function: { name: 'lookup' } },
        reasoning: 'low',
        onPayload: (payload) => {
          request = payload
        },
      },
    )
    return { message, request }
  })
  await info.attach('named-tool-choice', {
    body: JSON.stringify(result, null, 2),
    contentType: 'application/json',
  })

  expect(result.request).toMatchObject({
    messages: expect.arrayContaining([
      {
        role: 'system',
        content: expect.stringContaining('Call only the lookup function.'),
      },
    ]),
  })
  expect(result.message.stopReason).toBe('toolUse')
  expect(result.message.content).toContainEqual(
    expect.objectContaining({ type: 'toolCall', name: 'lookup', arguments: { code: 'A7' } }),
  )
})

test('Pi keeps Qwen3.5 tool names that contain spaces', async ({ page }, info) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const result = await page.evaluate(async () => {
    const { models, model } = await window.piChecks.provider(true)
    const agent = await window.piChecks.createAgent(true)
    const tool = { ...agent.state.tools[0]!, name: 'read file', label: 'Read file' }
    let request: unknown
    const message = await models.complete(
      model,
      {
        systemPrompt: 'Use read file when asked to look up a code. Report its result.',
        tools: [tool],
        messages: [
          { role: 'user', content: 'Use read file to look up code A7.', timestamp: Date.now() },
        ],
      },
      {
        toolChoice: { type: 'function', function: { name: 'read file' } },
        reasoning: 'low',
        onPayload: (payload) => {
          request = payload
        },
      },
    )
    return { message, request }
  })
  await info.attach('spaced-tool-name', {
    body: JSON.stringify(result, null, 2),
    contentType: 'application/json',
  })

  expect(result.request).toMatchObject({
    messages: expect.arrayContaining([
      {
        role: 'system',
        content: expect.stringContaining('Call only the read file function.'),
      },
    ]),
  })
  expect(result.message.stopReason).toBe('toolUse')
  expect(result.message.content).toContainEqual(
    expect.objectContaining({ type: 'toolCall', name: 'read file', arguments: { code: 'A7' } }),
  )
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
  test(
    `${modelId}: bare JSON mode returns JSON with thinking off`,
    modelCheck(modelId),
    async ({ page }) => {
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
              {
                role: 'user',
                content: 'Return a JSON object with answer equal to 6.',
                timestamp: 0,
              },
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
      expect(result.payload).toMatchObject({
        response_format: { type: 'json_object', schema: '{}' },
      })
      expect(result.native.length).toBeGreaterThan(0)
      const prefix = '<think>\n\n</think>\n\n'
      expect(result.text).toBe(
        result.native.startsWith(prefix) ? result.native.slice(prefix.length) : result.native,
      )
      expect(JSON.parse(result.text)).toEqual({ answer: 6 })
    },
  )
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

const modelIds = process.env.MODEL_IDS?.split(',') ?? [
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
]
const checks = modelIds.flatMap((modelId) =>
  (modelId === 'Qwen2.5-7B-Instruct-q4f16_1-MLC' ? [false, true] : [false]).map(
    (constrainedToolCalls) => ({ modelId, constrainedToolCalls }),
  ),
)
for (const { modelId, constrainedToolCalls } of checks) {
  const protocol = modelId === 'Qwen2.5-7B-Instruct-q4f16_1-MLC' && !constrainedToolCalls
  const title = protocol
    ? 'Pi Agent Core preserves native tool output'
    : 'Pi Agent Core tools, usage, cancellation and recovery in a Worker'
  const feature = constrainedToolCalls ? 'constrained-tools' : 'tools'
  test(
    `${modelId}: ${title}` + (constrainedToolCalls ? ' with constraints' : ''),
    modelCheck(
      modelId,
      protocol ? 'tool-protocol' : feature,
      ...(!modelId.startsWith('Qwen2.5-') ? (['thinking'] as const) : []),
    ),
    async ({ page }, info) => {
      const errors: string[] = []
      page.on('pageerror', (error) => errors.push(error.message))
      await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
      await page.evaluate((id) => window.piChecks.load(id), modelId)
      const evidence = await page.evaluate(
        async ({ modelId, constrainedToolCalls }) => {
          const agent = await window.piChecks.createAgent(true, { constrainedToolCalls })
          const requests: ChatCompletionRequestStreaming[] = []
          const native: WebLLMChunk[] = []
          let toolCalls = 0
          agent.subscribe((event) => {
            if (event.type === 'tool_execution_start') toolCalls++
          })
          agent.onPayload = (payload) => {
            requests.push(structuredClone(payload) as ChatCompletionRequestStreaming)
          }
          agent.onProviderStreamEvent = (event) => {
            native.push(structuredClone(event) as WebLLMChunk)
          }
          const error = await agent
            .prompt(
              modelId === 'Qwen3.5-0.8B-q4f16_1-MLC'
                ? 'Call lookup with code A7 and report exactly the answer it returns.'
                : 'Look up A7 and tell me the answer.',
            )
            .then(
              () => null,
              (error: unknown) => String(error),
            )
          let direct: string | undefined
          if (!constrainedToolCalls && agent.state.errorMessage) {
            const { engine } = await window.piChecks.provider(true)
            direct = await window.piChecks.withEngine(engine, async (engine) => {
              await engine.resetChat()
              let text = ''
              for await (const chunk of await engine.chat.completions.create(requests[0]!))
                text += chunk.choices[0]?.delta.content ?? ''
              return text
            })
          }
          return { messages: agent.state.messages, requests, native, error, direct, toolCalls }
        },
        { modelId, constrainedToolCalls },
      )
      const path = info.outputPath('model.json')
      await writeFile(path, JSON.stringify(evidence, null, 2))
      await info.attach('messages', {
        path,
        contentType: 'application/json',
      })
      expect(evidence.error).toBeNull()
      const { messages } = evidence
      const last = messages.at(-1)
      if (last?.role === 'assistant' && last.stopReason === 'error') {
        expect(protocol).toBe(true)
        expect(last.errorMessage).toContain('Qwen returned invalid tool JSON.')
        expect(evidence.toolCalls).toBe(0)
        expect(messages.some((message) => message.role === 'toolResult')).toBe(false)
        expect(last.content.some((part) => part.type === 'toolCall')).toBe(false)
        expect(evidence.requests).toHaveLength(1)
        expect(evidence.requests[0]).not.toHaveProperty('response_format')
        const raw = evidence.native
          .flatMap((chunk) =>
            chunk.choices.map((choice) =>
              'delta' in choice ? (choice.delta.content ?? '') : choice.text,
            ),
          )
          .join('')
        expect(evidence.direct?.startsWith(raw)).toBe(true)
        expect(raw.match(/<tool_call>/g)).toHaveLength(1)
        expect(raw.match(/<\/tool_call>/g)).toHaveLength(1)
        const source = raw.match(/<tool_call>([\s\S]*)<\/tool_call>/)?.[1]
        expect(source).toBeDefined()
        expect(() => JSON.parse(source!)).toThrow(SyntaxError)
        return
      }
      if (constrainedToolCalls)
        expect(evidence.requests[0]?.response_format?.type).toBe('structural_tag')
      else expect(evidence.requests[0]).not.toHaveProperty('response_format')
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
        const text = final.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join('')
        const thinking = messages
          .filter((message) => message.role === 'assistant')
          .map((message) =>
            message.content
              .filter((part) => part.type === 'thinking')
              .map((part) => part.thinking)
              .join(''),
          )
        const prefix = modelId.startsWith('Qwen3.5-')
          ? '<think>'
          : evidence.requests.map((request) => {
              const prompt = 'prompt' in request ? request.prompt : undefined
              return typeof prompt === 'string' && prompt.endsWith('<|channel>thought\n')
                ? '<|channel>thought\n'
                : ''
            })
        expect(text.trim()).not.toBe('')
        expect({ text, thinking }).toEqual(nativeOutput(evidence.native, modelId, prefix))
        expect(final.usage.totalTokens).toBeGreaterThan(0)
      }
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
    },
  )
}

const gemmaModel = 'gemma-4-E2B-it-q4f16_1-MLC'

test(
  `${gemmaModel}: preserves thinking history when sampling sets WebLLM options`,
  modelCheck(gemmaModel),
  async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), gemmaModel)
    const evidence = await page.evaluate(async () => {
      const { models, model } = await window.piChecks.provider(true)
      const preserved = 'Prior lookup reasoning should survive.'
      const history: AssistantMessage = {
        role: 'assistant',
        api: 'webllm',
        provider: 'webllm',
        model: model.id,
        content: [
          { type: 'thinking', thinking: preserved },
          { type: 'toolCall', id: 'lookup-1', name: 'lookup', arguments: { code: 'A7' } },
        ],
        stopReason: 'toolUse',
        timestamp: 0,
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
      }
      let payload: unknown
      const message = await models.complete(
        model,
        {
          messages: [
            { role: 'user', content: 'Look up A7.', timestamp: 0 },
            history,
            {
              role: 'toolResult',
              toolCallId: 'lookup-1',
              toolName: 'lookup',
              content: [{ type: 'text', text: 'ORCHID-42' }],
              isError: false,
              timestamp: 0,
            },
            {
              role: 'user',
              content: 'What was the lookup result? Reply with only the value.',
              timestamp: 0,
            },
          ],
        },
        {
          request: { extra_body: { enable_thinking: true, preserve_thinking: true } },
          samplingParams: {
            extra_body: { enable_thinking: true, enable_latency_breakdown: true },
          },
          onPayload: (request) => {
            payload = structuredClone(request)
          },
        },
      )
      return { message, payload, preserved }
    })

    expect(evidence.payload).toMatchObject({
      extra_body: { enable_latency_breakdown: true },
      prompt: expect.stringContaining(`<|think|>\n`),
    })
    expect(evidence.payload).not.toHaveProperty('extra_body.enable_thinking')
    expect(evidence.payload).not.toHaveProperty('extra_body.preserve_thinking')
    expect(evidence.payload).toMatchObject({ prompt: expect.stringContaining(evidence.preserved) })
    expect(evidence.message.stopReason).toBe('stop')
    expect(
      evidence.message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    ).toContain('ORCHID-42')
  },
)

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
    const [, thinking, text] = /^([\s\S]*?)<\/think>([\s\S]*)$/.exec(raw) ?? []
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
          const options: SimpleStreamOptions = {
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
      const output = enable_thinking && modelId.startsWith('Qwen3.5-') ? '<think>' + direct : direct
      expect(text, mode).toBe(output.replace(/^<think>[\s\S]*?<\/think>/, ''))
      const thinking = message.content
        .filter((part) => part.type === 'thinking')
        .map((part) => part.thinking)
        .join('')
      const nativeThinking = /<think>([\s\S]*?)<\/think>/.exec(output)?.[1] ?? ''
      expect(thinking, mode).toBe(nativeThinking)
      if (enable_thinking && modelId.startsWith('Qwen3-')) expect(thinking, mode).toMatch(/\S/)
      if (!enable_thinking) expect(thinking, mode).toMatch(/^\s*$/)
    }
  })
}
