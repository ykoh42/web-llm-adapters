import { afterEach, expect, it, vi } from 'vitest'
import {
  Agent,
  AudioBlock,
  CachePointBlock,
  ContextWindowOverflowError,
  GuardContentBlock,
  Message,
  TextBlock,
} from '@strands-agents/sdk'
import { BrowserModel, WebLLMModel, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from './helpers.js'

afterEach(() => vi.unstubAllGlobals())
const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

function open() {
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
  return { ...native, model, bridge, request }
}

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
  const { model, engine, create } = open()
  const prompt = [
    new TextBlock('One'),
    new CachePointBlock({ cacheType: 'default' }),
    new TextBlock('Two'),
  ]
  const options = {
    systemPrompt: prompt,
    toolSpecs: [{ name: 'read', description: 'Read a file' }],
  }
  const native = new WebLLMModel({
    engine,
    modelId: 'test-model',
  })
  expect(await model.countTokens(messages, options)).toBe(
    await native.countTokens(messages, { ...options, systemPrompt: 'One\n\nTwo' }),
  )
  await collect(model.stream(messages, { systemPrompt: prompt }))
  expect(create.mock.calls[0]?.[0].messages[0]).toEqual({ role: 'system', content: 'One\n\nTwo' })
})

it('rejects unsupported system blocks instead of dropping them at the boundary', async () => {
  const { model, create } = open()
  const systemPrompt = [new GuardContentBlock({ text: { text: 'Protected text', qualifiers: [] } })]
  await expect(collect(model.stream(messages, { systemPrompt }))).rejects.toThrow(
    'guardContentBlock',
  )
  await expect(model.countTokens(messages, { systemPrompt })).rejects.toThrow('guardContentBlock')
  expect(create).not.toHaveBeenCalled()
})

it('preserves audio bytes across JSON without changing the native request', async () => {
  const { model, create } = open()
  const audio = [
    new Message({
      role: 'user',
      content: [
        new AudioBlock({ format: 'wav', source: { bytes: new Uint8Array([0, 127, 128, 255]) } }),
      ],
    }),
  ]
  await collect(model.stream(audio))
  expect(create.mock.calls[0]?.[0].messages[0]).toMatchObject({
    content: [
      { type: 'text', text: '' },
      { type: 'input_audio', input_audio: { format: 'wav', data: 'AH+A/w==' } },
    ],
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
  create.mockResolvedValueOnce(
    (async function* () {
      try {
        yield chunk({ content: 'Partial' })
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

it('cleans up after early consumer return', async () => {
  const { model, stop } = open()
  const stream = model.stream(messages)
  await stream.next()
  await stream.return(undefined)
  expect(stop).toHaveBeenCalledTimes(1)
  await expect(collect(model.stream(messages))).resolves.toBeDefined()
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
  await expect(bridge.count(request)).rejects.toThrow('closed')
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
  await expect(collect(model.stream(messages))).resolves.toBeDefined()
})

it('forwards tool choices and trailing context without sharing per-call options', async () => {
  const { model, bridge, create } = open()
  const start = vi.spyOn(bridge, 'start')
  const toolSpecs = [{ name: 'read', description: 'Read a file' }]
  const toolChoice = { tool: { name: 'read' } }
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
