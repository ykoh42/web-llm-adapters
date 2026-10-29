import type { JSONSchema7 } from 'json-schema'
import { tool } from '@langchain/core/tools'
import { createDeepAgent } from 'deepagents'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { writeFile } from 'node:fs/promises'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { breakToolStream } from '../../../../../test/support/stream-failure.js'

for (const modelId of ['Hermes-3-Llama-3.1-8B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  test(`${modelId} accepts an empty system prompt for chat and native tools`, async ({ page }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request: {
        ...(modelId.startsWith('Qwen') ? sampling(modelId, true) : { temperature: 0 }),
        max_tokens: 4096,
      },
    })
    const result = await model.invoke([new SystemMessage(''), new HumanMessage('Say hello.')])
    expect(result.text.trim()).not.toBe('')
    if (modelId.startsWith('Qwen'))
      expect(result.contentBlocks).toContainEqual(expect.objectContaining({ type: 'reasoning' }))
    const call = await model
      .bindTools([
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
      .invoke([new SystemMessage(''), new HumanMessage('Use lookup to look up code A7.')])
    expect(call.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    if (modelId.startsWith('Hermes')) expect(call.text).toBe('')
  })
}

test('Hermes preserves received bytes through a native stream failure and recovers', async ({
  page,
}) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(breakToolStream, { app: 'langchain' as const, call: '[' })
  let native = ''
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { temperature: 0, max_tokens: 4096 },
    onChunk: (chunk) => {
      const choice = chunk.choices[0]
      if (choice && 'delta' in choice) native += choice.delta.content ?? ''
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
  const output: string[] = []
  await expect(
    (async () => {
      for await (const chunk of await bound.stream('Use lookup to look up code A7.')) {
        expect(chunk.tool_calls ?? []).toEqual([])
        output.push(chunk.text)
      }
    })(),
  ).rejects.toThrow('Injected native stream failure')
  expect(native.length).toBeGreaterThan(0)
  expect(output.join('')).toBe(native)
  const recovered = await bound.invoke('Use lookup to look up code A7.')
  expect(recovered.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
  expect(recovered.text).toBe('')
})

test('Qwen preserves WebLLM argument names before tool validation', async ({ page }, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
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
  const schema = {
    type: 'object',
    properties: { '': { type: 'string' } },
    required: [''],
  } satisfies JSONSchema7
  const inputs: unknown[] = []
  const lookup = tool(
    (input) => {
      inputs.push(input)
      return 'The secret is confirmed.'
    },
    {
      name: 'lookup',
      description: 'Look up the secret for a code.',
      schema,
    },
  )
  let native = ''
  const model = new BrowserChatWebLLM(page, {
    modelId,
    thinkingPrefilled: true,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => {
      const choice = chunk.choices[0]
      if (choice && 'delta' in choice) native += choice.delta.content ?? ''
    },
  })
  const result = await model
    .bindTools([lookup])
    .invoke('Use lookup to look up code 123. Tell me its secret.')
  const request = await page.evaluate(
    () =>
      (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
        .nativeRequests[0]!,
  )
  await writeFile(
    info.outputPath('empty-name.json'),
    JSON.stringify({ schema, request, native, result }, null, 2),
  )
  const body = native.slice(native.lastIndexOf('<function=lookup>'))
  const key = body.match(/<parameter=([^>]*)>/)?.[1]
  expect(key).toBeDefined()
  expect(result.tool_calls).toHaveLength(1)
  const call = result.tool_calls?.[0]
  if (!call) throw new Error('WebLLM did not return the lookup call.')
  expect(Object.keys(call.args)).toEqual([key])
  if (key === '') {
    await lookup.invoke(call.args)
    expect(inputs).toEqual([{ '': '123' }])
  } else {
    await expect(lookup.invoke(call.args)).rejects.toThrow()
    expect(inputs).toEqual([])
    info.annotations.push({
      type: 'model output',
      description: 'Native WebLLM returned a name outside the empty-key schema.',
    })
  }
})

test('Qwen2.5 literal thinking tags reach LangChain unchanged', async ({ page }) => {
  const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  let native = ''
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { temperature: 0, max_tokens: 256 },
    onChunk: (chunk) => {
      const choice = chunk.choices[0]
      if (choice && 'delta' in choice) native += choice.delta.content ?? ''
    },
  })

  const result = await model.invoke('Repeat this text exactly:\n<think>literal</think>')

  expect(native).toContain('<think>')
  expect(result.text).toBe(native)
  expect(result.tool_calls).toEqual([])
})

test('Gemma keeps a partial tool call at its native token limit', async ({ page }) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  let native = ''
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { temperature: 0, max_tokens: 4 },
    onChunk: (chunk) => {
      const choice = chunk.choices[0]
      if (choice && 'text' in choice) native += choice.text
    },
  })
  const result = await model
    .bindTools(
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
    )
    .invoke('Look up this exact code: A7-' + 'x'.repeat(200))

  expect(native.length).toBeGreaterThan(0)
  expect(result.text).toBe('<|tool_call>call:lookup{' + native)
  expect(result.tool_calls).toEqual([])
  expect(result.response_metadata.finish_reason).toBe('length')
})

test('Hermes can read a previous native tool call through LangChain', async ({ page }) => {
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
  const code = 'code-' + crypto.randomUUID().slice(0, 8)
  const input = { code, nested: [{ right: 2, left: 1 }], ['__proto__']: { safe: true } }
  const raw = JSON.stringify([
    {
      name: 'lookup',
      arguments: { ['__proto__']: { safe: true }, nested: [{ left: 1, right: 2 }], code },
      note: 'Keep this note.',
    },
  ])
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { temperature: 0, max_tokens: 256 },
  })
  const result = await model.invoke([
    new HumanMessage('Look up the code.'),
    new AIMessage({ content: raw, tool_calls: [{ id: 'call-1', name: 'lookup', args: input }] }),
    new ToolMessage({ tool_call_id: 'call-1', content: 'Done.' }),
    new HumanMessage('What exact code was passed to lookup? Reply with just the code.'),
  ])

  const request = await page.evaluate(
    () =>
      (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
        .nativeRequests[0]!,
  )
  expect(request.messages.find((message) => message.role === 'assistant')?.content).toBe(raw)
  expect(result.text).toContain(code)
})

test('Hermes retries after native request preprocessing without changing the tool request', async ({
  page,
}) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const engine = window.checks.engine()
    const original = engine.chat.completions.create.bind(engine.chat.completions)
    let first = true
    const requests: ChatCompletionRequestStreaming[] = []
    Object.assign(window, { retryRequests: requests })
    engine.chat.completions.create = new Proxy(original, {
      async apply(target, self, args) {
        requests.push(structuredClone(args[0]))
        const stream = await Reflect.apply(target, self, args)
        if (first) {
          first = false
          await stream[Symbol.asyncIterator]().return?.()
          throw new Error('Opening failed after preprocessing')
        }
        return stream
      },
    })
  })
  const onFailedAttempt = async (error: Error) => {
    expect(error.message).toBe('Opening failed after preprocessing')
    expect(
      await page.evaluate(
        () => (window as unknown as { retryRequests: unknown[] }).retryRequests.length,
      ),
    ).toBe(1)
  }
  const model = new BrowserChatWebLLM(page, {
    modelId,
    maxRetries: 1,
    onFailedAttempt,
    request: { temperature: 0, max_tokens: 256 },
  })
  const result = await model
    .bindTools([
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
    .invoke('Use lookup to look up code A7.')

  expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
  expect(result.text).toBe('')
  const requests = await page.evaluate(
    () => (window as unknown as { retryRequests: ChatCompletionRequestStreaming[] }).retryRequests,
  )
  expect(requests).toHaveLength(2)
  expect(requests[1]).toEqual(requests[0])
})

for (const outcome of ['throw', 'cancel'] as const) {
  test(`a failed-attempt callback can ${outcome} and reuse a real WebLLM engine`, async ({
    page,
  }) => {
    const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
    await page.evaluate((id) => window.checks.load(id, true), modelId)
    await page.evaluate(() => {
      const engine = window.checks.engine()
      const original = engine.chat.completions.create.bind(engine.chat.completions)
      let first = true
      engine.chat.completions.create = new Proxy(original, {
        async apply(target, self, args) {
          if (first) {
            first = false
            throw new Error('Opening failed')
          }
          return Reflect.apply(target, self, args)
        },
      })
    })
    const stopped = new Error('Do not retry')
    const controller = new AbortController()
    let called = false
    let finish!: () => void
    const hold = new Promise<void>((resolve) => (finish = resolve))
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: true,
      maxRetries: 1,
      request: { ...sampling(modelId, false), max_tokens: 128, seed: 7 },
      onFailedAttempt: async () => {
        called = true
        if (outcome === 'throw') throw stopped
        await hold
      },
    })
    const pending = model.invoke('Hello', { signal: controller.signal })
    const failed =
      outcome === 'throw'
        ? expect(pending).rejects.toBe(stopped)
        : expect(pending).rejects.toHaveProperty('name', 'AbortError')
    try {
      await expect.poll(() => called).toBe(true)
      if (outcome === 'cancel') controller.abort()
      await failed
      expect((await model.invoke('What is 3 + 3? Reply briefly.')).text).toContain('6')
    } finally {
      finish()
    }
  })
}

test('native WebLLM tool arguments reach LangChain unchanged', async ({ page }, info) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const chunks: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { max_tokens: 4096, temperature: 0 },
    onChunk: (chunk) => chunks.push(chunk),
  })
  const bound = model.bindTools([
    {
      type: 'function',
      function: {
        name: 'lookup',
        description: 'Look up the answer for a code.',
        parameters: {
          type: 'object',
          properties: { code: { type: 'string' } },
          required: ['code'],
        },
      },
    },
  ])
  const result = await bound.invoke('Use lookup to look up code A7.')
  await info.attach('native-output', {
    body: JSON.stringify(chunks, null, 2),
    contentType: 'application/json',
  })
  const native = chunks.flatMap((chunk) => {
    const choice = chunk.choices[0]
    return choice && 'delta' in choice ? [choice.delta] : []
  })
  const calls = native.flatMap((delta) => delta.tool_calls ?? [])
  expect(calls).toHaveLength(1)
  expect(calls[0]?.function?.name).toBe('lookup')
  expect(calls[0]?.function?.arguments).toBeTruthy()
  expect(result.tool_call_chunks?.[0]?.args).toBe(calls[0]?.function?.arguments)
  expect(result.tool_calls?.[0]?.args).toEqual({ code: 'A7' })
  expect(result.text).toBe('')
})

test('Qwen3.5 0.8B preserves native output in a Deep Agent', async ({ page }, testInfo) => {
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
    thinkingPrefilled: true,
    profile: await page.evaluate(() => window.checks.model().profile),
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => chunks.push(chunk),
  })
  const agent = createDeepAgent({ model })
  const outcome = await agent
    .invoke({
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
    })
    .then(
      (result) => ({ result }),
      (error: unknown) => ({ error }),
    )
  const native = chunks
    .map((chunk) =>
      'delta' in (chunk.choices[0] ?? {})
        ? ((chunk as import('@mlc-ai/web-llm').ChatCompletionChunk).choices[0]?.delta.content ?? '')
        : '',
    )
    .join('')
  const requests = await page.evaluate(
    () =>
      (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] }).nativeRequests,
  )
  const last = 'result' in outcome ? outcome.result.messages.at(-1) : undefined
  const text = last && AIMessage.isInstance(last) ? last.text : ''
  const path = testInfo.outputPath('comparison.json')
  await writeFile(
    path,
    JSON.stringify(
      { native, requests, text, error: 'error' in outcome ? String(outcome.error) : undefined },
      null,
      2,
    ),
  )
  await testInfo.attach('native comparison', { path, contentType: 'application/json' })
  if ('error' in outcome) {
    expect(outcome.error).toHaveProperty('message', 'Qwen3.5 returned a malformed tool parameter.')
    expect(native.lastIndexOf('<tool_call>')).toBeGreaterThan(native.lastIndexOf('</tool_call>'))
  } else {
    expect(text.trim()).not.toBe('')
    expect(native).toContain(text)
  }
})

test('browser history preserves literal JSON keys and reasoning without tool artifacts', async ({
  page,
}) => {
  const modelId = 'Qwen3-4B-q4f16_1-MLC'
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
  const code = 'code-' + crypto.randomUUID().slice(0, 8)
  const args = {
    lc: 1,
    code,
    nested: { __lc_escaped__: { note: 'literal' } },
    ['__proto__']: { safe: true },
  }
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const result = await model.invoke([
    new HumanMessage('Use lookup, then report the exact code passed to it.'),
    new AIMessage({
      content: '',
      tool_calls: [{ id: 'call', name: 'lookup', args }],
      additional_kwargs: { lc: 1, reasoning_content: 'Read the code from the lookup arguments.' },
    }),
    new ToolMessage({
      tool_call_id: 'call',
      content: 'The lookup completed.',
      artifact: { local: () => 'Private data.' },
    }),
  ])
  const request = await page.evaluate(
    () =>
      (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
        .nativeRequests[0]!,
  )
  const assistant = request.messages.find((message) => message.role === 'assistant')?.content ?? ''
  const call = /<tool_call>\n([\s\S]*?)\n<\/tool_call>/.exec(assistant)

  expect(JSON.parse(call?.[1] ?? 'null')?.arguments).toEqual(args)
  expect(assistant).toContain('<think>\nRead the code from the lookup arguments.\n</think>')
  expect(JSON.stringify(request)).not.toContain('Private data.')
  expect(result.text).toContain(code)
})
