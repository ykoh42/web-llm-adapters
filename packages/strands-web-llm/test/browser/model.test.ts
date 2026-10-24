import { afterEach, expect, it, vi } from 'vitest'
import {
  Agent,
  AudioBlock,
  CachePointBlock,
  ContextWindowOverflowError,
  GuardContentBlock,
  Message,
  ModelError,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge,
  type BrowserPage,
} from '../../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())
const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

it('delivers native tool chunks while the model is still generating arguments', async () => {
  const { page, bridge, create } = open()
  const hold = pause()
  const onChunk = vi.fn()
  create.mockResolvedValueOnce(
    (async function* () {
      yield chunk({ content: '<tool_call>' })
      yield chunk({ content: '{"name":"lookup","arguments":{' })
      await hold.wait
      yield chunk({ content: '}}</tool_call>' }, 'stop')
    })(),
  )
  const model = new BrowserModel(page, { modelId: 'Qwen3-8B-q4f16_1-MLC', onChunk })
  const pending = collect(
    model.stream(messages, { toolSpecs: [{ name: 'lookup', description: 'Look up a note.' }] }),
  )
  try {
    await vi.waitFor(() => expect(onChunk).toHaveBeenCalledTimes(2))
  } finally {
    hold.resume()
    await pending
    await bridge.close()
  }
  expect(onChunk).toHaveBeenCalledTimes(3)
  expect(await pending).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
})

it.each(['abort', 'throw'])(
  'closes a pending native tool stream when its callback %s',
  async (mode) => {
    const { page, bridge, create, stop, model: plain } = open()
    const hold = pause()
    const closed = vi.fn()
    const controller = new AbortController()
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: '<tool_call>' })
          await hold.wait
          yield chunk({ content: '{"name":"lookup","arguments":{}}</tool_call>' }, 'stop')
        } finally {
          closed()
        }
      })(),
    )
    stop.mockImplementation(async () => hold.resume())
    const onChunk = vi.fn(() => {
      if (mode === 'abort') controller.abort()
      else throw new Error('Callback failed')
    })
    const model = new BrowserModel(page, { modelId: 'Qwen3-8B-q4f16_1-MLC', onChunk })
    const pending = collect(
      model.stream(messages, {
        toolSpecs: [{ name: 'lookup', description: 'Look up a note.' }],
        cancelSignal: controller.signal,
      }),
    )
    try {
      if (mode === 'abort') await expect(pending).rejects.toHaveProperty('name', 'AbortError')
      else await expect(pending).rejects.toThrow('Callback failed')
      expect(onChunk).toHaveBeenCalledTimes(1)
      expect(closed).toHaveBeenCalledTimes(1)
      expect(stop).toHaveBeenCalledTimes(1)
      expect(await collect(plain.stream(messages))).toContainEqual({
        type: 'modelMessageStopEvent',
        stopReason: 'endTurn',
      })
    } finally {
      hold.resume()
      await bridge.close()
    }
  },
)

it('stops a native chunk batch when its callback cancels the request', async () => {
  const { page, bridge } = open()
  const controller = new AbortController()
  const onChunk = vi.fn(() => controller.abort())
  vi.spyOn(bridge, 'next').mockResolvedValueOnce({
    done: false,
    value: { type: 'modelMessageStartEvent', role: 'assistant' },
    chunks: [chunk({ content: 'First' }), chunk({ content: 'Late' }, 'stop')],
  })
  const model = new BrowserModel(page, { modelId: 'test-model', onChunk })
  const output: unknown[] = []
  await expect(
    (async () => {
      for await (const event of model.stream(messages, { cancelSignal: controller.signal }))
        output.push(event)
    })(),
  ).rejects.toHaveProperty('name', 'AbortError')
  expect(onChunk).toHaveBeenCalledTimes(1)
  expect(output).toEqual([])
  await bridge.close()
})

it('transfers provider content without session metadata or tracking IDs', async () => {
  const { model, bridge } = open()
  const start = vi.spyOn(bridge, 'start')
  const metadata = { custom: { private: 'session-only', score: Infinity } }
  const message = new Message({
    role: 'user',
    content: [new TextBlock('Hello')],
    metadata,
  })

  await collect(model.stream([message]))

  expect(start.mock.calls[0]?.[0].messages).toEqual([
    { role: 'user', content: [{ text: 'Hello' }] },
  ])
  expect(message.metadata).toBe(metadata)
  expect(message.trackingId).toBeTruthy()
  await bridge.close()
})

function open() {
  vi.stubGlobal('scheduler', { yield: () => new Promise<void>((resolve) => setImmediate(resolve)) })
  const native = setup()
  native.create.mockImplementation(async () => chunks(chunk({ content: 'Hi' }, 'stop')))
  const bridge = createBrowserBridge(native.engine)
  vi.stubGlobal('window', { webllmBridge: bridge })
  // Exercise the JSON boundary in both directions, as a browser transport does.
  const page: BrowserPage = {
    evaluate: async (run, arg) => {
      const result = await run(arg)
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  const config = { modelId: 'test-model', maxTokens: 128, contextWindowLimit: 8192 }
  const model = new BrowserModel(page, config)
  const request = { messages: messages.map((message) => message.toJSON()), config, options: {} }
  return { ...native, model, bridge, request, page }
}

it('keeps overlapping bridge reads in event order without duplicating events', async () => {
  const { bridge, request } = open()
  const id = await bridge.start(request)
  try {
    await bridge.next(id)
    const results = await Promise.all([bridge.next(id), bridge.next(id), bridge.next(id)])
    expect(
      results.map((result) => {
        if ('error' in result) throw new Error(result.error.message)
        return result.done ? 'done' : result.value?.type
      }),
    ).toEqual([
      'modelContentBlockStartEvent',
      'modelContentBlockDeltaEvent',
      'modelContentBlockStopEvent',
    ])
  } finally {
    await bridge.stop(id)
  }
})

it('delivers every native chunk once when bridge reads overlap', async () => {
  const { bridge, request, create } = open()
  const first = chunk({ content: 'Hi' })
  const last = chunk({}, 'stop')
  create.mockResolvedValueOnce(chunks(first, last))
  const id = await bridge.start({ ...request, includeChunks: true })
  try {
    const results = await Promise.all(Array.from({ length: 12 }, () => bridge.next(id)))
    expect(results.flatMap((result) => result.chunks ?? [])).toEqual([first, last])
    const events = results.flatMap((result) => {
      if ('error' in result) throw new Error(result.error.message)
      return result.done || 'native' in result ? [] : [result.value.type]
    })
    expect(events).toEqual([
      'modelMessageStartEvent',
      'modelContentBlockStartEvent',
      'modelContentBlockDeltaEvent',
      'modelContentBlockStopEvent',
      'modelMessageStopEvent',
    ])
  } finally {
    await bridge.stop(id)
  }
})

it('prepares one stream event while the browser consumer handles the previous one', async () => {
  const { bridge, request, create } = open()
  const produced = vi.fn()
  create.mockResolvedValueOnce(
    (async function* () {
      produced('first')
      yield chunk({ content: 'First' })
      produced('second')
      yield chunk({ content: 'Second' }, 'stop')
    })(),
  )
  const id = await bridge.start(request)
  try {
    expect(await bridge.next(id)).toMatchObject({
      value: { type: 'modelMessageStartEvent' },
    })
    await vi.waitFor(() => expect(produced).toHaveBeenCalledExactlyOnceWith('first'))
    await Promise.resolve()
    expect(produced).toHaveBeenCalledTimes(1)
  } finally {
    await bridge.stop(id)
  }
})

it('preserves Strands model errors across the browser boundary', async () => {
  const { model, create } = open()
  const invalid = chunk({ content: 'Invalid' }, 'stop')
  invalid.choices[0]!.index = 1
  create.mockResolvedValueOnce(chunks(invalid))
  await expect(collect(model.stream(messages))).rejects.toBeInstanceOf(ModelError)
  await expect(collect(model.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
})

it('delivers buffered native output before a browser stream failure', async () => {
  const { model, create, bridge } = open()
  const raw = '[{"name":"lookup","arguments":{'
  create.mockResolvedValueOnce(
    (async function* () {
      yield chunk({ content: raw })
      throw new TypeError('Native stream failed')
    })(),
  )
  const output: string[] = []
  await expect(
    (async () => {
      for await (const event of model.stream(messages, {
        toolSpecs: [{ name: 'lookup', description: 'Look up a code.' }],
      })) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          output.push(event.delta.text)
        if (event.type === 'modelContentBlockStartEvent') expect(event.start).toBeUndefined()
      }
    })(),
  ).rejects.toBeInstanceOf(TypeError)
  expect(output.join('')).toBe(raw)
  await expect(collect(model.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
  await bridge.close()
})

it('preserves model error causes across the browser boundary', async () => {
  const { model, create } = open()
  model.updateConfig({ modelId: 'Qwen3-4B-q4f16_1-MLC' })
  create.mockResolvedValueOnce(
    chunks(chunk({ content: '<tool_call>{"name":"lookup","arguments": }</tool_call>' }, 'stop')),
  )
  const failed = collect(
    model.stream(messages, {
      toolSpecs: [{ name: 'lookup', description: 'Look up a value.' }],
    }),
  )
  await expect(failed).rejects.toMatchObject({
    name: 'ModelError',
    cause: { name: 'WebLLMError', cause: expect.any(SyntaxError) },
  })
})

it('keeps the stream error when browser cleanup also fails', async () => {
  const failure = new ModelError('Generation failed')
  let call = 0
  const page: BrowserPage = {
    async evaluate<T>(): Promise<T> {
      call++
      if (call === 1) return 'stream-id' as T
      if (call === 2) throw failure
      throw new Error('Page closed during cleanup')
    },
  }
  const model = new BrowserModel(page, { modelId: 'test-model' })

  await expect(collect(model.stream(messages))).rejects.toBe(failure)
})

it('does not send a redundant stop after the browser stream ends', async () => {
  const { model, bridge } = open()
  const stop = vi.spyOn(bridge, 'stop')

  await collect(model.stream(messages))

  expect(stop).not.toHaveBeenCalled()
})

it('delivers native chunks with logprobs through a Strands Agent in order', async () => {
  const { page, create, model } = open()
  const first = chunk({ content: 'Hi' })
  first.choices[0]!.logprobs = {
    content: [{ token: 'Hi', logprob: -0.25, bytes: [72, 105], top_logprobs: [] }],
  }
  const last = chunk({}, 'stop')
  create.mockResolvedValueOnce(chunks(first, last))
  const onChunk = vi.fn()
  const browser = new BrowserModel(page, { ...model.getConfig(), onChunk })
  const result = await new Agent({ model: browser, printer: false }).invoke('Hello')
  expect(result.lastMessage.content).toEqual([new TextBlock('Hi')])
  expect(onChunk.mock.calls.map(([value]) => value)).toEqual([first, last])
  expect(browser.getConfig()).not.toHaveProperty('onChunk')
})

it('exposes log probabilities in the browser stop event', async () => {
  const { page, bridge, create } = open()
  const first = chunk({ content: 'Hi' })
  first.choices[0]!.logprobs = {
    content: [{ token: 'Hi', logprob: -0.25, bytes: [72, 105], top_logprobs: [] }],
  }
  const last = chunk({}, 'stop')
  last.choices[0]!.logprobs = {
    content: [{ token: '!', logprob: -0.5, bytes: [33], top_logprobs: [] }],
  }
  create.mockResolvedValueOnce(chunks(first, last))
  const model = new BrowserModel(page, {
    modelId: 'test-model',
    request: { logprobs: true },
  })

  try {
    const events = await collect(model.stream(messages))
    const stop = events.find((event) => event.type === 'modelMessageStopEvent')
    expect(stop).toMatchObject({
      additionalModelResponseFields: {
        logprobs: {
          content: [
            { token: 'Hi', logprob: -0.25, bytes: [72, 105], top_logprobs: [] },
            { token: '!', logprob: -0.5, bytes: [33], top_logprobs: [] },
          ],
        },
      },
    })
  } finally {
    await bridge.close()
  }
})

it('delivers the native chunk that caused a protocol error before reporting it', async () => {
  const { page, create, model } = open()
  const invalid = chunk({ content: 'Hi' }, 'stop')
  invalid.choices[0]!.index = 1
  create.mockResolvedValueOnce(chunks(invalid))
  const onChunk = vi.fn()
  const browser = new BrowserModel(page, { ...model.getConfig(), onChunk })
  await expect(collect(browser.stream(messages))).rejects.toThrow('one response')
  expect(onChunk).toHaveBeenCalledExactlyOnceWith(invalid)
  await expect(collect(browser.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
})

it('closes the native stream when the Node.js chunk callback throws', async () => {
  const { page, model, create, stop } = open()
  const closed = vi.fn()
  create.mockResolvedValueOnce(
    (async function* () {
      try {
        yield chunk({ content: 'Hi' })
      } finally {
        closed()
      }
    })(),
  )
  const browser = new BrowserModel(page, {
    ...model.getConfig(),
    onChunk: () => {
      throw new Error('Callback failed')
    },
  })
  await expect(collect(browser.stream(messages))).rejects.toThrow('Callback failed')
  expect(closed).toHaveBeenCalledOnce()
  expect(stop).toHaveBeenCalledOnce()
  await expect(collect(model.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
})

it('does not transfer native chunks unless requested', async () => {
  const { bridge, request } = open()
  const id = await bridge.start(request)
  while (true) {
    const result = await bridge.next(id)
    expect(result).not.toHaveProperty('chunks')
    if ('error' in result) throw new Error(result.error.message)
    if (result.done) break
  }
})

it('snapshots tool history and definitions before a delayed page transfer', async () => {
  const { create, bridge } = open()
  const hold = pause()
  const submitted = pause()
  const page: BrowserPage = {
    async evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T> {
      submitted.resume()
      await hold.wait
      return run(structuredClone(arg))
    },
  }
  const model = new BrowserModel(page, { modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' })
  const input = { city: 'Before' }
  const toolSpecs = [{ name: 'weather', description: 'Before' }]
  const history = [
    ...messages,
    new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ toolUseId: 'call-1', name: 'weather', input })],
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
  ]
  const pending = collect(model.stream(history, { toolSpecs }))
  try {
    await submitted.wait
    input.city = 'After'
    toolSpecs[0]!.description = 'After'
    hold.resume()
    await pending
    expect(create.mock.calls[0]?.[0].messages[1]).toMatchObject({
      role: 'assistant',
      content: '[{"name":"weather","arguments":{"city":"Before"}}]',
    })
    expect(create.mock.calls[0]?.[0].tools?.[0]?.function.description).toBe('Before')
  } finally {
    hold.resume()
    await pending.catch(() => {})
    await bridge.close()
  }
})

it('runs a Strands Agent through the public browser bridge and keeps config snapshots', async () => {
  const { model, create, engine } = open()
  const unload = vi.spyOn(engine, 'unload')
  const settings = { seed: 7 }
  model.updateConfig({ request: settings })
  settings.seed = 9
  const snapshot = model.getConfig()
  snapshot.request!.seed = 11
  expect(model.getConfig().request?.seed).toBe(7)
  const result = await new Agent({ model, printer: false }).invoke('Hello')
  expect(result.lastMessage.content).toEqual([new TextBlock('Hi')])
  expect(create.mock.calls[0]?.[0]).toMatchObject({ max_tokens: 128, seed: 7 })
  expect(unload).not.toHaveBeenCalled()
})

it('counts the browser prompt, joins system text and ignores cache hints', async () => {
  const { model, engine, create, page } = open()
  const prompt = [
    new TextBlock('One'),
    new CachePointBlock({ cacheType: 'default' }),
    new TextBlock('Two'),
  ]
  const options = {
    systemPrompt: prompt,
    toolSpecs: [{ name: 'read', description: 'Read a file' }],
  }
  const evaluate = vi.spyOn(page, 'evaluate')
  const native = new WebLLMModel({
    engine,
    modelId: 'test-model',
  })
  expect(await model.countTokens(messages, options)).toBe(
    await native.countTokens(messages, { ...options, systemPrompt: 'OneTwo' }),
  )
  expect(evaluate).not.toHaveBeenCalled()
  await collect(model.stream(messages, { systemPrompt: prompt }))
  expect(create.mock.calls[0]?.[0].messages[0]).toEqual({ role: 'system', content: 'OneTwo' })
})

it('counts Gemma prompts locally without crossing the browser bridge', async () => {
  const { model, engine, page } = open()
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  const request = { extra_body: { enable_thinking: true } }
  model.updateConfig({ modelId, request })
  const native = new WebLLMModel({ engine, modelId, request })
  const evaluate = vi.spyOn(page, 'evaluate')
  const options = {
    systemPrompt: 'Think carefully.',
    toolSpecs: [{ name: 'read', description: 'Read a file' }],
  }

  await expect(model.countTokens(messages, options)).resolves.toBe(
    await native.countTokens(messages, options),
  )
  expect(evaluate).not.toHaveBeenCalled()
})

it('counts system guards and rejects them from WebLLM requests', async () => {
  const { model, create, engine } = open()
  const systemPrompt = [new GuardContentBlock({ text: { text: 'Protected text', qualifiers: [] } })]
  const native = new WebLLMModel({ engine, modelId: 'test-model' })
  await expect(model.countTokens(messages, { systemPrompt })).resolves.toBe(
    await native.countTokens(messages, { systemPrompt }),
  )
  await expect(collect(model.stream(messages, { systemPrompt }))).rejects.toThrow(
    'guardContentBlock',
  )
  expect(create).not.toHaveBeenCalled()
})

it('preserves audio bytes across the browser boundary without changing the native request', async () => {
  const { model, create, page } = open()
  const evaluate = vi.spyOn(page, 'evaluate')
  const audio = [
    new Message({
      role: 'user',
      content: [
        new AudioBlock({ format: 'wav', source: { bytes: new Uint8Array([0, 127, 128, 255]) } }),
      ],
    }),
  ]
  await collect(model.stream(audio))
  expect(JSON.parse(String(evaluate.mock.calls[0]?.[1]))).toMatchObject({
    messages: [{ content: [{ audio: { source: { bytes: 'AH+A/w==' } } }] }],
  })
  expect(create.mock.calls[0]?.[0].messages[0]).toMatchObject({
    content: [{ type: 'input_audio', input_audio: { format: 'wav', data: 'AH+A/w==' } }],
  })
})

it.each(['overflow', 'native', 'string'] as const)(
  'forwards %s errors and allows the next request',
  async (kind) => {
    const { model, create } = open()
    const error =
      kind === 'string'
        ? 'Worker failed'
        : Object.assign(new Error('Request failed'), {
            name: kind === 'overflow' ? 'ContextWindowSizeExceededError' : 'NativeError',
          })
    create.mockRejectedValueOnce(error)
    const failed = collect(model.stream(messages))
    if (kind === 'overflow') await expect(failed).rejects.toBeInstanceOf(ContextWindowOverflowError)
    else
      await expect(failed).rejects.toMatchObject({
        name: kind === 'native' ? 'NativeError' : 'Error',
      })
    await expect(collect(model.stream(messages))).resolves.toContainEqual({
      type: 'modelMessageStopEvent',
      stopReason: 'endTurn',
    })
  },
)

it('stops an abandoned browser iterator on abort and releases the shared engine', async () => {
  const { model, create, stop } = open()
  const closed = pause()
  const running = pause()
  // Keep inference active until interrupted, even when the bridge reads one event ahead.
  stop.mockImplementationOnce(async () => running.resume())
  create.mockResolvedValueOnce(
    (async function* () {
      try {
        yield chunk({ content: 'Partial' })
        await running.wait
      } finally {
        closed.resume()
      }
    })(),
  )
  const controller = new AbortController()
  const first = model.stream(messages, { cancelSignal: controller.signal })
  await first.next()
  await first.next()
  await first.next()
  controller.abort(new Error('Stopped'))
  await closed.wait
  expect(stop).toHaveBeenCalledTimes(1)
  await expect(collect(model.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
  await first.return(undefined)
})

it('does not start a pre-cancelled request', async () => {
  const { model, bridge } = open()
  const start = vi.spyOn(bridge, 'start')
  await expect(
    collect(model.stream(messages, { cancelSignal: AbortSignal.abort(new Error('Stopped')) })),
  ).rejects.toThrow('Stopped')
  expect(start).not.toHaveBeenCalled()
})

it('explains how to set up the browser bridge when it is missing', async () => {
  const { model } = open()
  vi.stubGlobal('window', {})
  await expect(collect(model.stream(messages))).rejects.toThrow(
    'Expose createBrowserBridge(engine) as window.webllmBridge before using BrowserModel.',
  )
})

it('cleans up after early consumer return', async () => {
  const { model, stop } = open()
  const stream = model.stream(messages)
  await stream.next()
  await stream.return(undefined)
  expect(stop).toHaveBeenCalledTimes(1)
  await expect(collect(model.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
})

it('waits for every stream to close and leaves engine unloading to the caller', async () => {
  const { model, engine, bridge, create, request } = open()
  const unload = vi.spyOn(engine, 'unload')
  const cleaning = pause()
  create.mockResolvedValueOnce(
    (async function* () {
      try {
        yield chunk({ content: 'Partial' })
      } finally {
        await cleaning.wait
      }
    })(),
  )
  const first = model.stream(messages)
  await first.next()
  await first.next()
  await first.next()
  const queued = await bridge.start(request)
  const closing = bridge.close()
  await expect(bridge.start(request)).rejects.toThrow('closed')
  cleaning.resume()
  await closing
  await bridge.stop(queued)
  await bridge.close()
  expect(unload).not.toHaveBeenCalled()
  await first.return(undefined)
  expect(await bridge.next(queued)).toMatchObject({ error: { message: 'Unknown browser call.' } })
})

it('keeps a failed stop visible to the consumer and still permits reuse', async () => {
  const { model, stop } = open()
  stop.mockRejectedValueOnce(new Error('Interrupt failed'))
  const first = model.stream(messages)
  await first.next()
  await expect(first.return(undefined)).rejects.toThrow('Interrupt failed')
  await expect(collect(model.stream(messages))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
})

it('forwards tool choices and trailing context without sharing per-call options', async () => {
  const { model, bridge, create } = open()
  const start = vi.spyOn(bridge, 'start')
  const toolSpecs = [{ name: 'read', description: 'Read a file' }]
  const toolChoice = { tool: { name: 'read' } }
  create.mockResolvedValueOnce(
    chunks(
      chunk(
        {
          tool_calls: [
            { index: 0, id: 'call', type: 'function', function: { name: 'read', arguments: '{}' } },
          ],
        },
        'tool_calls',
      ),
    ),
  )
  await collect(model.stream(messages, { toolSpecs, toolChoice, dynamicTrailingBlocks: 2 }))
  expect(start.mock.calls[0]?.[0].options).toEqual({
    toolSpecs,
    toolChoice,
    dynamicTrailingBlocks: 2,
  })
  expect(create.mock.calls[0]?.[0]).toMatchObject({
    tool_choice: { type: 'function', function: { name: 'read' } },
  })
  await collect(model.stream(messages))
  expect(start.mock.calls[1]?.[0].options).toEqual({})
})

it('uses native APIs through page evaluation and waits for them during bridge close', async () => {
  const { bridge, engine, page } = open()
  const stats = vi.spyOn(engine, 'runtimeStatsText').mockResolvedValue('native stats')
  expect(
    await page.evaluate(
      () => window.webllmBridge.withEngine((engine) => engine.runtimeStatsText()),
      undefined,
    ),
  ).toBe('native stats')
  expect(stats).toHaveBeenCalledOnce()
  const entered = pause()
  const finish = pause()
  const active = bridge.withEngine(async () => {
    entered.resume()
    await finish.wait
    return 'finished'
  })
  await entered.wait
  const run = vi.fn()
  const queued = expect(bridge.withEngine(run)).rejects.toThrow('closed')
  let closed = false
  const closing = bridge.close().then(() => {
    closed = true
  })
  await expect(bridge.withEngine(run)).rejects.toThrow('closed')
  expect(closed).toBe(false)
  finish.resume()
  expect(await active).toBe('finished')
  await queued
  await closing
  expect(closed).toBe(true)
  expect(run).not.toHaveBeenCalled()
})

it('preserves reserved JSON keys when browser RPC filters object properties', async () => {
  const { model, page, create } = open()
  page.evaluate = async (run, arg) =>
    run(JSON.parse(JSON.stringify(arg), (key, value) => (key === '__proto__' ? undefined : value)))
  const input = JSON.parse('{"__proto__":{"safe":true},"lc":1}')
  const history = [
    new Message({ role: 'user', content: [new TextBlock('Look up the data.')] }),
    new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input })],
    }),
    new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'call',
          status: 'success',
          content: [new TextBlock('Done')],
        }),
      ],
    }),
  ]

  await collect(model.stream(history))

  const assistant = create.mock.calls[0]?.[0].messages[1]
  if (assistant?.role !== 'assistant') throw new Error('Missing assistant history.')
  expect(JSON.parse(assistant.tool_calls![0]!.function.arguments)).toEqual(input)
})

it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
  'rejects invalid browser input numbers instead of changing them to null: %s',
  async (temperature) => {
    const { model, page, create } = open()
    const evaluate = vi.spyOn(page, 'evaluate')
    model.updateConfig({ temperature })

    await expect(collect(model.stream(messages))).rejects.toThrow('finite numbers')

    expect(evaluate).not.toHaveBeenCalled()
    expect(create).not.toHaveBeenCalled()
  },
)

it('matches direct Qwen3.5 history when optional arguments are undefined', async () => {
  const { model, page, engine, create } = open()
  model.updateConfig({ modelId: 'Qwen3.5-4B-q4f16_1-MLC' })
  const direct = new WebLLMModel({ engine, ...model.getConfig() })
  const input = { code: 'A7', empty: '', zero: 0, disabled: false, nullable: null }
  Object.assign(input, { optional: undefined })
  const history = [
    new Message({ role: 'user', content: [new TextBlock('Look up the code.')] }),
    new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input })],
    }),
    new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'call',
          status: 'success',
          content: [new TextBlock('Done')],
        }),
      ],
    }),
  ]

  await collect(model.stream(history))
  await collect(direct.stream(history))

  expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
  expect(Object.hasOwn(input, 'optional')).toBe(true)
  await page.evaluate(() => window.webllmBridge.close(), undefined)
})
