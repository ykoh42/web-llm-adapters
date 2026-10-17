import { describe, expect, it, vi } from 'vitest'
import {
  Agent,
  ContextWindowOverflowError,
  ContentBlockEvent,
  FunctionTool,
  MaxTokensError,
  Message,
  TextBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import type { ModelStreamEvent, StreamOptions } from '@strands-agents/sdk'
import { WebLLMModel } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from './helpers.js'

const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]
const tools = [{ name: 'weather', description: 'Get the weather' }]

async function aggregate(model: WebLLMModel, options: StreamOptions = {}) {
  const stream = model.streamAggregated(messages, options)
  for (;;) {
    const step = await stream.next()
    if (step.done) return step.value
  }
}

describe('WebLLMModel', () => {
  it('can disable tools for a call while retaining the Agent tool registry', async () => {
    const { model, create } = setup(chunk({ content: 'Hello' }, 'stop'))
    model.updateConfig({ request: { tool_choice: 'none' } })
    const callback = vi.fn(() => 'unused')
    const agent = new Agent({
      model,
      printer: false,
      tools: [
        new FunctionTool({
          name: 'weather',
          description: 'Get the weather',
          callback,
        }),
      ],
    })
    await agent.invoke('Say hello without using tools.')
    expect(callback).not.toHaveBeenCalled()
    expect(create.mock.calls[0]![0].tool_choice).toBe('none')
    expect(create.mock.calls[0]![0]).not.toHaveProperty('tools')
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Hello' }, 'stop')))
    await collect(model.stream(messages, { toolSpecs: tools, toolChoice: { auto: {} } }))
    expect(create.mock.calls[1]![0].tool_choice).toBe('auto')
    expect(create.mock.calls[1]![0].tools).toHaveLength(1)
  })
  it('lets a completed-block hook use the same engine before the outer stream returns', async () => {
    const { model, engine, create } = setup(chunk({ content: 'First' }, 'stop'))
    create.mockResolvedValueOnce(chunks(chunk({ content: 'First' }, 'stop')))
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Second' }, 'stop')))
    const aux = new WebLLMModel({ engine, modelId: 'test-model' })
    const agent = new Agent({ model, printer: false })
    agent.addHook(ContentBlockEvent, async () => {
      const result = await aggregate(aux)
      expect(result.message.content).toEqual([new TextBlock('Second')])
    })
    expect(String(await agent.invoke('Hi'))).toBe('First')
  })
  it.each(['native', 'worker', 'stream'])(
    'maps %s context overflow for Strands recovery',
    async (kind) => {
      const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
      const error =
        kind === 'worker'
          ? 'ContextWindowSizeExceededError: Prompt tokens exceed context window size'
          : Object.assign(new Error('Prompt tokens exceed context window size'), {
              name: 'ContextWindowSizeExceededError',
            })
      if (kind === 'stream')
        create.mockResolvedValueOnce(
          (async function* () {
            yield chunk({})
            throw error
          })(),
        )
      else create.mockRejectedValueOnce(error)
      await expect(collect(model.stream(messages))).rejects.toBeInstanceOf(
        ContextWindowOverflowError,
      )
      expect(await collect(model.stream(messages))).toContainEqual(
        expect.objectContaining({ delta: { type: 'textDelta', text: 'Recovered' } }),
      )
    },
  )

  it('streams text through a real Strands Agent', async () => {
    const { model, create } = setup(chunk({ content: 'Hello' }), chunk({}, 'stop'))
    const agent = new Agent({ model, printer: false })
    const result = await agent.invoke('Hi')
    expect(result.stopReason).toBe('endTurn')
    expect(result.lastMessage?.content).toEqual([new TextBlock('Hello')])
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'test-model',
        stream: true,
        stream_options: { include_usage: true },
        n: 1,
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    )
  })

  it('emits text before waiting for the rest of the response', async () => {
    const { model, create } = setup()
    const hold = pause()
    let finished = false
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'Hi' })
        await hold.wait
        finished = true
        yield chunk({ content: ' there' }, 'stop')
      })(),
    )
    const stream = model.stream(messages)
    try {
      expect((await stream.next()).value).toMatchObject({ type: 'modelMessageStartEvent' })
      expect((await stream.next()).value).toMatchObject({ type: 'modelContentBlockStartEvent' })
      expect((await stream.next()).value).toMatchObject({ delta: { text: 'Hi' } })
      expect(finished).toBe(false)
      hold.resume()
      const rest = await collect(stream)
      expect(rest).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'textDelta', text: ' there' },
      })
      expect(rest.at(-1)).toEqual({ type: 'modelMessageStopEvent', stopReason: 'endTurn' })
    } finally {
      hold.resume()
      await stream.return(undefined)
    }
  })

  it('keeps native request options and raw chunks available', async () => {
    const part = chunk({ content: 'Hi' }, 'stop')
    const onChunk = vi.fn()
    const { engine, create } = setup(part)
    const model = new WebLLMModel({
      engine,
      modelId: 'test-model',
      maxTokens: 64,
      temperature: 0,
      topP: 0.9,
      onChunk,
      request: { seed: 7, logprobs: true, repetition_penalty: 1.1 },
    })
    await collect(model.stream(messages))
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        seed: 7,
        logprobs: true,
        repetition_penalty: 1.1,
        max_tokens: 64,
        temperature: 0,
        top_p: 0.9,
      }),
    )
    expect(onChunk).toHaveBeenCalledWith(part)
    expect(model.engine).toBe(engine)
  })

  it('copies config so WebLLM cannot mutate the next request', () => {
    const { engine } = setup()
    const request = { temperature: 0.5 }
    const model = new WebLLMModel({ engine, modelId: 'first', request })
    request.temperature = 1
    const copy = model.getConfig()
    copy.request!.temperature = 2
    expect(model.getConfig().request?.temperature).toBe(0.5)
    model.updateConfig({ modelId: 'second', maxTokens: 10 })
    expect(model.getConfig()).toMatchObject({ modelId: 'second', maxTokens: 10 })
    model.updateConfig({ temperature: 0.2 })
    expect(model.getConfig().temperature).toBe(0.2)
  })

  it('keeps the final usage chunk after the stop choice', async () => {
    const usage = {
      prompt_tokens: 5,
      completion_tokens: 2,
      total_tokens: 7,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 10,
        decode_tokens_per_s: 2,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.5,
      },
    }
    const { model } = setup(chunk({ content: 'Hi' }, 'stop'), {
      ...chunk(),
      choices: [],
      usage,
    })
    const result = await aggregate(model)
    expect(result.metadata?.usage).toEqual({ inputTokens: 5, outputTokens: 2, totalTokens: 7 })
    expect(result.metadata?.trace).toEqual(usage.extra)
  })

  it('runs a tool and sends its result back through the Agent loop', async () => {
    const { model, create } = setup()
    create.mockResolvedValueOnce(
      chunks(
        chunk({ content: '[{"name":"weather","arguments":{}}]' }),
        chunk(
          {
            tool_calls: [
              { index: 0, id: 'call-1', function: { name: 'weather', arguments: '{}' } },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Sunny' }, 'stop')))
    const weather = new FunctionTool({ ...tools[0]!, callback: () => 'Sunny' })
    const agent = new Agent({ model, tools: [weather], printer: false })
    const result = await agent.invoke('Weather?')
    expect(result.lastMessage?.content).toEqual([new TextBlock('Sunny')])
    expect(create).toHaveBeenCalledTimes(2)
    const request = create.mock.calls[1]![0]
    expect(request.messages).toContainEqual({
      role: 'tool',
      tool_call_id: 'call-1',
      content: 'Sunny',
    })
    const assistant = request.messages.find((message) => message.role === 'assistant')
    expect(assistant).toMatchObject({ content: null, tool_calls: [{ id: 'call-1' }] })
  })

  it('joins split tool names and JSON without showing raw tool JSON as text', async () => {
    const { model } = setup(
      chunk({
        content: 'raw tool JSON',
        tool_calls: [{ index: 0, id: 'a', function: { name: 'wea', arguments: '{"city":' } }],
      }),
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'ther', arguments: '"Seoul"}' } }] },
        'tool_calls',
      ),
    )
    const stream = model.streamAggregated(messages, { toolSpecs: tools })
    const blocks = []
    for await (const event of stream) {
      if (event instanceof TextBlock || event instanceof ToolUseBlock) blocks.push(event)
    }
    expect(blocks).toEqual([
      new ToolUseBlock({ name: 'weather', toolUseId: 'a', input: { city: 'Seoul' } }),
    ])
  })

  it('never turns a truncated tool call into executable tool use', async () => {
    const { model } = setup(
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{' } }] },
        'length',
      ),
    )
    await expect(aggregate(model, { toolSpecs: tools })).rejects.toBeInstanceOf(MaxTokensError)
  })

  it('rejects broken tool JSON', async () => {
    const { model } = setup(
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{' } }] },
        'tool_calls',
      ),
    )
    await expect(aggregate(model, { toolSpecs: tools })).rejects.toThrow('invalid tool JSON')
  })

  it('rejects a stream without a stop reason', async () => {
    const { model } = setup(chunk({ content: 'cut off' }))
    await expect(aggregate(model)).rejects.toThrow('without a stop reason')
  })

  it('does not invent tools when the backend returns none', async () => {
    const { model } = setup(chunk({}, 'tool_calls'))
    await expect(aggregate(model)).rejects.toThrow('no tool calls')
  })

  it('rejects a tool choice WebLLM cannot represent', async () => {
    const { model, create } = setup()
    await expect(collect(model.stream(messages, { toolChoice: { any: {} } }))).rejects.toThrow(
      'toolChoice.any',
    )
    expect(create).not.toHaveBeenCalled()
  })

  it('does not start an already cancelled call', async () => {
    const { model, create, stop } = setup()
    await expect(
      collect(model.stream(messages, { cancelSignal: AbortSignal.abort() })),
    ).rejects.toThrow()
    expect(create).not.toHaveBeenCalled()
    expect(stop).not.toHaveBeenCalled()
  })

  it('interrupts active generation and closes its stream when cancelled', async () => {
    const { model, create, stop } = setup()
    const hold = pause()
    const waiting = pause()
    let closed = false
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: 'Hi' })
          waiting.resume()
          await hold.wait
          yield chunk({}, 'abort')
        } finally {
          closed = true
        }
      })(),
    )
    stop.mockImplementation(async () => {
      hold.resume()
    })
    const controller = new AbortController()
    const stream = model.stream(messages, { cancelSignal: controller.signal })
    await stream.next()
    await stream.next()
    expect((await stream.next()).value).toMatchObject({ delta: { text: 'Hi' } })
    const rest = collect(stream)
    await waiting.wait
    controller.abort()
    try {
      expect(stop).toHaveBeenCalledTimes(1)
      const events = await rest
      expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
      expect(closed).toBe(true)
    } finally {
      hold.resume()
      await stream.return(undefined)
    }
  })

  it('removes the cancel listener after a completed call', async () => {
    const { model, stop } = setup(chunk({ content: 'Hi' }, 'stop'))
    const controller = new AbortController()
    await collect(model.stream(messages, { cancelSignal: controller.signal }))
    controller.abort()
    expect(stop).not.toHaveBeenCalled()
  })

  it('releases a cancelled stream even when the Agent leaves its iterator suspended', async () => {
    const { model, create } = setup()
    let closed = false
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: 'Partial' })
        } finally {
          closed = true
        }
      })(),
    )
    const controller = new AbortController()
    const agent = new Agent({ model, printer: false })
    for await (const update of agent.stream('Hi', { cancelSignal: controller.signal })) {
      if (
        update.type === 'modelStreamUpdateEvent' &&
        update.event.type === 'modelContentBlockDeltaEvent'
      )
        controller.abort()
    }
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
    const next = await new Agent({ model, printer: false }).invoke('Try again')
    expect(next.lastMessage.content).toEqual([new TextBlock('Recovered')])
    expect(closed).toBe(true)
  })

  it('waits for native cleanup before reusing an abandoned cancelled stream', async () => {
    const { model, create } = setup()
    const closing = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: 'Partial' })
        } finally {
          await closing.wait
        }
      })(),
    )
    const controller = new AbortController()
    const first = model.stream(messages, { cancelSignal: controller.signal })
    await first.next()
    await first.next()
    await first.next()
    controller.abort()
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
    const next = aggregate(model)
    expect(create).toHaveBeenCalledTimes(1)
    closing.resume()
    await expect(next).resolves.toMatchObject({ stopReason: 'endTurn' })
    await first.return(undefined)
  })

  it('closes a stream cancelled while the backend is still opening it', async () => {
    const { model, create } = setup()
    const opening = pause()
    const started = pause()
    const close = vi.fn(async () => ({ done: true as const, value: undefined }))
    create.mockImplementationOnce(async () => {
      started.resume()
      await opening.wait
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: false as const, value: chunk({}, 'stop') }),
          return: close,
        }),
      }
    })
    const controller = new AbortController()
    const response = collect(model.stream(messages, { cancelSignal: controller.signal }))
    await started.wait
    controller.abort()
    opening.resume()
    await expect(response).rejects.toThrow()
    expect(close).toHaveBeenCalledTimes(1)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Hi' }, 'stop')))
    await expect(aggregate(model)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('closes the native stream when a chunk callback fails', async () => {
    const { engine, create } = setup()
    let closed = false
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: 'Hi' })
        } finally {
          closed = true
        }
      })(),
    )
    const model = new WebLLMModel({
      engine,
      modelId: 'test-model',
      onChunk: () => {
        throw new Error('Callback failed')
      },
    })
    await expect(aggregate(model)).rejects.toThrow('Callback failed')
    expect(closed).toBe(true)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Hi' }, 'stop')))
    const next = new WebLLMModel({ engine, modelId: 'test-model' })
    await expect(aggregate(next)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('does not start generation when cancelled after the message start', async () => {
    const { model, stop } = setup(chunk({ content: 'Hi' }), chunk({}, 'stop'))
    const controller = new AbortController()
    const stream = model.stream(messages, { cancelSignal: controller.signal })
    await stream.next()
    controller.abort()
    expect(await collect(stream)).toEqual([
      { type: 'modelMessageStopEvent', stopReason: 'cancelled' },
    ])
    expect(stop).toHaveBeenCalledTimes(1)
  })

  it('stops and releases the engine when a consumer leaves early', async () => {
    const { model, stop, create } = setup(chunk({ content: 'Hi' }), chunk({}, 'stop'))
    const stream = model.stream(messages)
    await stream.next()
    await stream.return(undefined)
    expect(stop).toHaveBeenCalledTimes(1)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Again' }, 'stop')))
    await expect(aggregate(model)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('runs overlapping adapters in order on one engine', async () => {
    const { model, engine, create, stop } = setup(chunk({ content: 'First' }, 'stop'))
    const first = model.stream(messages)
    await first.next()
    const second = new WebLLMModel({ engine, modelId: 'other' })
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Second' }, 'stop')))
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Third' }, 'stop')))
    const next = aggregate(second)
    const last = aggregate(model)
    expect(create).toHaveBeenCalledTimes(1)
    await collect(first)
    expect((await next).message.content).toEqual([new TextBlock('Second')])
    expect((await last).message.content).toEqual([new TextBlock('Third')])
    expect(stop).not.toHaveBeenCalled()
  })

  it('cancels a queued call without interrupting active work or letting later calls overtake it', async () => {
    const { model, create, stop } = setup(chunk({ content: 'First' }, 'stop'))
    const first = model.stream(messages)
    await first.next()
    const controller = new AbortController()
    const waiting = collect(model.stream(messages, { cancelSignal: controller.signal }))
    const cancelled = expect(waiting).rejects.toThrow()
    controller.abort()
    await cancelled
    expect(stop).not.toHaveBeenCalled()
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Last' }, 'stop')))
    const last = aggregate(model)
    expect(create).toHaveBeenCalledTimes(1)
    await collect(first)
    expect((await last).message.content).toEqual([new TextBlock('Last')])
    expect(create).toHaveBeenCalledTimes(2)
    expect(stop).not.toHaveBeenCalled()
  })

  it('keeps engines independent while another engine has an active call', async () => {
    const first = setup(chunk({ content: 'First' }, 'stop'))
    const second = setup(chunk({ content: 'Other engine' }, 'stop'))
    const active = first.model.stream(messages)
    await active.next()
    try {
      expect((await aggregate(second.model)).message.content).toEqual([
        new TextBlock('Other engine'),
      ])
    } finally {
      await active.return(undefined)
    }
  })

  it('releases queued work after the previous backend fails to open', async () => {
    const { model, create } = setup()
    const opening = pause()
    create.mockImplementationOnce(async () => {
      await opening.wait
      throw new Error('GPU lost')
    })
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
    const first = aggregate(model)
    const failed = expect(first).rejects.toThrow('GPU lost')
    const second = aggregate(model)
    opening.resume()
    await failed
    expect((await second).message.content).toEqual([new TextBlock('Recovered')])
  })

  it('releases the engine after a backend error', async () => {
    const { model, create } = setup()
    create.mockRejectedValueOnce(new Error('GPU lost'))
    await expect(aggregate(model)).rejects.toThrow('GPU lost')
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
    await expect(aggregate(model)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('closes a failed stream and allows the next call on the engine', async () => {
    const { model, create } = setup()
    let closed = false
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: 'Partial' })
          throw new Error('GPU lost during generation')
        } finally {
          closed = true
        }
      })(),
    )
    await expect(aggregate(model)).rejects.toThrow('GPU lost during generation')
    expect(closed).toBe(true)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
    await expect(aggregate(model)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('keeps interleaved tool calls and their results separate', async () => {
    const { model, create } = setup()
    const weather = vi.fn((input: unknown) => JSON.stringify(input))
    const time = vi.fn((input: unknown) => JSON.stringify(input))
    create.mockResolvedValueOnce(
      chunks(
        chunk({
          tool_calls: [
            { index: 0, id: 'weather-1', function: { name: 'weather', arguments: '{"city":' } },
          ],
        }),
        chunk({
          tool_calls: [
            { index: 1, id: 'time-1', function: { name: 'time', arguments: '{"zone":' } },
          ],
        }),
        chunk({ tool_calls: [{ index: 0, function: { arguments: '"Seoul"}' } }] }),
        chunk(
          { tool_calls: [{ index: 1, function: { arguments: '"Asia/Seoul"}' } }] },
          'tool_calls',
        ),
      ),
    )
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Done' }, 'stop')))
    const agent = new Agent({
      model,
      printer: false,
      tools: [
        new FunctionTool({ name: 'weather', description: 'Get weather', callback: weather }),
        new FunctionTool({ name: 'time', description: 'Get time', callback: time }),
      ],
    })
    expect((await agent.invoke('Weather and time?')).lastMessage.content).toEqual([
      new TextBlock('Done'),
    ])
    expect(weather.mock.calls).toHaveLength(1)
    expect(time.mock.calls).toHaveLength(1)
    expect(weather.mock.calls[0]![0]).toEqual({ city: 'Seoul' })
    expect(time.mock.calls[0]![0]).toEqual({ zone: 'Asia/Seoul' })
    expect(create.mock.calls[1]![0].messages).toEqual(
      expect.arrayContaining([
        { role: 'tool', tool_call_id: 'weather-1', content: '{"city":"Seoul"}' },
        { role: 'tool', tool_call_id: 'time-1', content: '{"zone":"Asia/Seoul"}' },
      ]),
    )
  })

  it('keeps the event order valid for an empty response', async () => {
    const { model } = setup(chunk({}, 'stop'))
    const events: ModelStreamEvent[] = await collect(model.stream(messages))
    expect(events).toEqual([
      { type: 'modelMessageStartEvent', role: 'assistant' },
      { type: 'modelMessageStopEvent', stopReason: 'endTurn' },
    ])
  })
})
