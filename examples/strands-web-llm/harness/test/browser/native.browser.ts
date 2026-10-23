import {
  Agent,
  Message,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
  type StreamOptions,
} from '@strands-agents/sdk'
import { test, expect } from '../../../fixtures/browser.js'
import { BrowserModel, pcmAudio, type WebLLMChunk } from 'strands-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { sampling } from '../../../sampling.js'
import { breakToolStream } from '../../../../../test/support/stream-failure.js'
import { recordNative, replayNative } from '../../fixtures/native.js'

test('Qwen preserves a caller stop inside thinking', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    const chunks: WebLLMChunk[] = []
    const settings = sampling(modelId, true)
    const model = new BrowserModel(page, {
      modelId,
      ...settings,
      maxTokens: 4096,
      request: { ...settings.request, seed: 7, stop: ['</think>'] },
      onChunk: (chunk) => chunks.push(chunk),
    })
    const stream = model.streamAggregated([
      new Message({ role: 'user', content: [new TextBlock('What is 17 * 23?')] }),
    ])
    let next = await stream.next()
    while (!next.done) next = await stream.next()
    const response = next.value
    const native = chunks
      .flatMap<WebLLMChunk['choices'][number]>((chunk) => chunk.choices)
      .map((choice) => ('delta' in choice ? (choice.delta.content ?? '') : choice.text))
      .join('')
    const reasoning = response.message.content
      .filter((block) => block.type === 'reasoningBlock')
      .map((block) => block.text ?? '')
      .join('')

    expect(response.stopReason).toBe('endTurn')
    expect(native).toMatch(/^<think>/)
    expect(native).not.toContain('</think>')
    expect(reasoning).not.toBe('')
    expect(reasoning).toBe(native.slice('<think>'.length))
    expect(response.message.content.every((block) => block.type === 'reasoningBlock')).toBe(true)
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

for (const modelId of ['Hermes-3-Llama-3.1-8B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  test(`${modelId} accepts an empty system prompt for chat and native tools`, async ({
    page,
  }, info) => {
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      await page.evaluate(() =>
        window.webllmBridge.withEngine((engine) => {
          window.emptySystemRequests = []
          const endpoint = engine.chat.completions
          endpoint.create = new Proxy(endpoint.create, {
            apply(target, self, args) {
              window.emptySystemRequests.push(structuredClone(args[0]))
              return Reflect.apply(target, self, args)
            },
          })
        }),
      )
      const chunks: WebLLMChunk[] = []
      const model = new BrowserModel(page, {
        modelId,
        ...(modelId.startsWith('Qwen') ? sampling(modelId, true) : { temperature: 0 }),
        maxTokens: 4096,
        onChunk: (chunk) => chunks.push(chunk),
      })
      let text = ''
      let reasoning = ''
      for await (const block of model.streamAggregated(
        [new Message({ role: 'user', content: [new TextBlock('Say hello.')] })],
        { systemPrompt: '' },
      )) {
        if (block instanceof TextBlock) text += block.text
        if (block.type === 'reasoningBlock') reasoning += block.text ?? ''
      }
      const request = await page.evaluate(() => window.emptySystemRequests[0]!)
      const nativeMessage = await page.evaluate(
        (id) => window.webllmBridge.withEngine((engine) => engine.getMessage(id)),
        modelId,
      )
      await recordNative(info, 'empty-system-chat', {
        request,
        nativeMessage,
        chunks,
        text,
        reasoning,
      })
      expect(request.messages[0]).toEqual({ role: 'system', content: '' })
      if (modelId.startsWith('Qwen')) {
        expect(request.extra_body?.enable_thinking).toBe(true)
        const thoughts = /<think>([\s\S]*?)<\/think>/g
        expect(reasoning).toBe(
          [...nativeMessage.matchAll(thoughts)].map((part) => part[1]).join(''),
        )
        expect(text).toBe(nativeMessage.replace(thoughts, ''))
      }
      expect(text.trim()).not.toBe('')
      const calls: ToolUseBlock[] = []
      for await (const block of model.streamAggregated(
        [new Message({ role: 'user', content: [new TextBlock('Use lookup to look up code A7.')] })],
        {
          systemPrompt: '',
          toolSpecs: [
            {
              name: 'lookup',
              description: 'Look up a code.',
              inputSchema: {
                type: 'object',
                properties: { code: { type: 'string' } },
                required: ['code'],
              },
            },
          ],
        },
      ))
        if (block instanceof ToolUseBlock) calls.push(block)
      expect(calls).toMatchObject([{ name: 'lookup', input: { code: 'A7' } }])
      await recordNative(info, 'empty-system-native-control', {
        request,
        native: await replayNative(page, [request]),
      })
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}

declare global {
  interface Window {
    emptySystemRequests: ChatCompletionRequestStreaming[]
  }
}

for (const modelId of ['gemma-4-E2B-it-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  test(`${modelId} keeps an empty user turn after real thinking`, async ({ page }) => {
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      await page.evaluate(() =>
        window.webllmBridge.withEngine((engine) => {
          Object.assign(window, { nativeRequests: [] })
          for (const endpoint of [engine.chat.completions, engine.completions])
            Object.assign(endpoint, {
              create: new Proxy(endpoint.create, {
                apply(target, self, args) {
                  ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
                    structuredClone(args[0]),
                  )
                  return Reflect.apply(target, self, args)
                },
              }),
            })
        }),
      )
      const settings = sampling(modelId, true)
      const model = new BrowserModel(page, {
        modelId,
        ...settings,
        maxTokens: 4096,
        request: { ...settings.request, seed: 7 },
      })
      const agent = new Agent({ model, printer: false })
      const first = await agent.invoke('What is 2 + 2? Answer briefly.')
      expect(
        first.lastMessage.content.some((block) => block.type === 'reasoningBlock' && block.text),
      ).toBe(true)
      const answer = first.lastMessage.content
        .filter((block) => block.type === 'textBlock')
        .map((block) => block.text)
        .join('')
      const saved = JSON.stringify(agent.messages)
      const stream = model.streamAggregated([
        ...agent.messages,
        new Message({ role: 'user', content: [] }),
      ])
      let stop: string | undefined
      for (;;) {
        const next = await stream.next()
        if (!next.done) continue
        stop = next.value.stopReason
        break
      }
      expect(stop).toBe('endTurn')
      const request = await page.evaluate(() =>
        (
          window as unknown as {
            nativeRequests: (ChatCompletionRequestStreaming | CompletionCreateParamsStreaming)[]
          }
        ).nativeRequests.at(-1)!,
      )
      if ('messages' in request) {
        expect(request.messages.at(-1)).toEqual({ role: 'user', content: '' })
        expect(request.messages.find((message) => message.role === 'assistant')?.content).toBe(
          answer,
        )
        expect(request.extra_body?.enable_thinking).toBe(true)
      } else {
        expect(request.prompt).toContain('<|turn>user\n<turn|>\n<|turn>model\n')
        expect(request.prompt).not.toContain('<|channel>thought\n')
        expect(request.prompt).toContain('<|think|>')
      }
      expect(JSON.stringify(agent.messages)).toBe(saved)
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}

test('Hermes preserves received bytes through a native stream failure and recovers', async ({
  page,
}) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    await page.evaluate(breakToolStream, { app: 'strands' as const, call: '[' })
    let native = ''
    const model = new BrowserModel(page, {
      modelId,
      maxTokens: 4096,
      temperature: 0,
      onChunk: (chunk) => {
        const choice = chunk.choices[0]
        if (choice && 'delta' in choice) native += choice.delta.content ?? ''
      },
    })
    const messages = [
      new Message({ role: 'user', content: [new TextBlock('Use lookup to look up code A7.')] }),
    ]
    const options: StreamOptions = {
      toolSpecs: [
        {
          name: 'lookup',
          description: 'Look up a code.',
          inputSchema: {
            type: 'object',
            properties: { code: { type: 'string' } },
            required: ['code'],
          },
        },
      ],
    }
    const output: string[] = []
    await expect(
      (async () => {
        for await (const event of model.stream(messages, options)) {
          if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
            output.push(event.delta.text)
          if (event.type === 'modelContentBlockStartEvent') expect(event.start).toBeUndefined()
        }
      })(),
    ).rejects.toThrow('Injected native stream failure')
    expect(native.length).toBeGreaterThan(0)
    expect(output.join('')).toBe(native)
    const recovered: ToolUseBlock[] = []
    for await (const event of model.streamAggregated(messages, options))
      if (event instanceof ToolUseBlock) recovered.push(event)
    expect(recovered).toMatchObject([{ name: 'lookup', input: { code: 'A7' } }])
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

test('Gemma returns constrained JSON, native logprobs and PCM results through BrowserModel', async ({
  page,
}) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    const chunks: WebLLMChunk[] = []
    const model = new BrowserModel(page, {
      modelId,
      maxTokens: 128,
      temperature: 0,
      onChunk: (chunk) => chunks.push(chunk),
      request: {
        extra_body: { enable_thinking: false },
        logprobs: true,
        top_logprobs: 2,
        response_format: {
          type: 'json_object',
          schema: JSON.stringify({
            type: 'object',
            properties: { word: { const: 'hello' } },
            required: ['word'],
          }),
        },
      },
    })
    const jsonRequest = model.getConfig().request!
    model.updateConfig({
      request: { response_format: { type: 'text' }, extra_body: { enable_thinking: true } },
      maxTokens: 512,
    })
    const agent = new Agent({ model, printer: false })
    const thought = await agent.invoke('What is 2 plus 2?')
    expect(thought.lastMessage.content.some((b) => b.type === 'reasoningBlock' && b.text)).toBe(
      true,
    )
    model.updateConfig({ request: jsonRequest, maxTokens: 128 })
    chunks.length = 0
    const result = await agent.invoke('Return JSON with word hello.')
    const text = result.lastMessage.content
      .filter((b) => b.type === 'textBlock')
      .map((b) => b.text)
      .join('')
    expect(JSON.parse(text)).toEqual({ word: 'hello' })
    expect(chunks.every((chunk) => chunk.object === 'chat.completion.chunk')).toBe(true)
    expect(
      chunks.some((chunk) => chunk.choices.some((choice) => choice.logprobs?.content?.length)),
    ).toBe(true)
    expect(chunks.at(-1)?.usage?.total_tokens).toBeGreaterThan(0)

    model.updateConfig({ request: { extra_body: { enable_thinking: false } } })
    chunks.length = 0
    const audio = pcmAudio(new Float32Array(16000), 16000)
    const heard = await new Agent({ model, printer: false }).invoke([
      new TextBlock('Describe any sound you hear in one short sentence.'),
      audio,
    ])
    expect(heard.stopReason).toBe('endTurn')
    expect(heard.lastMessage.content.some((b) => b.type === 'textBlock' && b.text.length)).toBe(
      true,
    )
    expect(chunks.at(-1)?.usage?.prompt_tokens).toBeGreaterThan(0)
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

test('Hermes can read a previous native tool call through Strands', async ({ page }) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  await page.evaluate(() =>
    window.webllmBridge.withEngine((engine) => {
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
    }),
  )
  try {
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
    const input = { code, nested: [{ right: 2, left: 1 }], ['__proto__']: { safe: true } }
    const raw = JSON.stringify([
      {
        name: 'lookup',
        arguments: { ['__proto__']: { safe: true }, nested: [{ left: 1, right: 2 }], code },
        note: 'Keep this note.',
      },
    ])
    const model = new BrowserModel(page, { modelId, temperature: 0, maxTokens: 256 })
    const history = [
      new Message({ role: 'user', content: [new TextBlock('Look up the code.')] }),
      new Message({
        role: 'assistant',
        content: [
          new TextBlock(raw),
          new ToolUseBlock({ toolUseId: 'call-1', name: 'lookup', input }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'call-1',
            status: 'success',
            content: [new TextBlock('Done.')],
          }),
        ],
      }),
      new Message({
        role: 'user',
        content: [new TextBlock('What exact code was passed to lookup? Reply with just the code.')],
      }),
    ]
    let output = ''
    for await (const event of model.streamAggregated(history)) {
      if (event instanceof TextBlock) output += event.text
    }
    const request = await page.evaluate(
      () =>
        (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
          .nativeRequests[0]!,
    )
    const assistant = request.messages.find((message) => message.role === 'assistant')
    expect(assistant?.content).toBe(raw)
    expect(JSON.parse(assistant?.content ?? 'null')[0]?.arguments).toEqual(input)
    expect(output).toContain(code)
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

test('Strands bare JSON mode preserves nested objects and root arrays', async ({ page }) => {
  const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    const model = new BrowserModel(page, {
      modelId,
      maxTokens: 128,
      temperature: 0,
      request: { response_format: { type: 'json_object' } },
    })
    const expected = [{ data: [1, true, null, { text: 'oak' }] }, [1, true, null, { text: 'oak' }]]
    for (const value of expected) {
      const agent = new Agent({ model, printer: false })
      const result = await agent.invoke('Return only this exact JSON: ' + JSON.stringify(value))
      expect(JSON.parse(String(result))).toEqual(value)
    }
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})
