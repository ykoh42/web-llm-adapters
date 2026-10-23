import { describe, expect, it, vi } from 'vitest'
import { Agent, Message, TextBlock, type StreamOptions } from '@strands-agents/sdk'
import { withEngine, type WebLLMModel } from '../src/index.js'
import { ModelError as ProtocolError } from '../../../shared/web-llm/types.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]
const tools = [{ name: 'weather', description: 'Get the weather' }]

async function aggregate(model: WebLLMModel, options: StreamOptions = {}) {
  const stream = model.streamAggregated(messages, options)
  for (;;) {
    const step = await stream.next()
    if (step.done) return step.value
  }
}

describe('cancellation', () => {
  it.each([
    Object.assign(new Error('Cancel the caller'), { name: 'ContextWindowSizeExceededError' }),
    'ContextWindowSizeExceededError: cancel the caller',
    new ProtocolError('Cancel the caller'),
  ])('keeps a queued cancellation reason without converting it: %s', async (reason) => {
    const { engine, model, create, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const started = pause()
    const hold = pause()
    const active = withEngine(engine, async () => {
      started.resume()
      await hold.wait
    })
    await started.wait
    const controller = new AbortController()
    const waiting = collect(model.stream(messages, { cancelSignal: controller.signal }))
    controller.abort(reason)
    try {
      await expect(waiting).rejects.toBe(reason)
      expect(create).not.toHaveBeenCalled()
      expect(stop).not.toHaveBeenCalled()
    } finally {
      hold.resume()
      await active
    }
    expect((await aggregate(model)).message.content).toEqual([new TextBlock('Recovered')])
  })

  it.each(['text', 'tool', 'native tool', 'metadata'])(
    'stops pending %s output when its consumer cancels',
    async (phase) => {
      const controller = new AbortController()
      const hasTool = phase === 'tool' || phase === 'native tool'
      const finish = hasTool ? 'tool_calls' : 'stop'
      const response = chunk(
        hasTool
          ? {
              tool_calls: [
                {
                  index: 0,
                  id: 'call-1',
                  function: { name: 'weather', arguments: '{"city":"Seoul"}' },
                },
              ],
            }
          : { content: 'Answer' },
        phase === 'native tool' ? null : finish,
      )
      response.usage = {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        extra: {
          e2e_latency_s: 1,
          prefill_tokens_per_s: 1,
          decode_tokens_per_s: 1,
          time_to_first_token_s: 1,
          time_per_output_token_s: 1,
        },
      }
      const { model, create } = setup()
      create.mockResolvedValueOnce(chunks(response))
      create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
      const after = []
      for await (const event of model.stream(messages, {
        toolSpecs: tools,
        cancelSignal: controller.signal,
      })) {
        if (controller.signal.aborted) after.push(event)
        if (
          (phase === 'metadata' && event.type === 'modelMetadataEvent') ||
          (phase !== 'metadata' && event.type === 'modelContentBlockStartEvent')
        )
          controller.abort()
      }
      expect(after).not.toContainEqual(
        expect.objectContaining({ type: 'modelContentBlockDeltaEvent' }),
      )
      expect(after.at(-1)).toEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
      expect((await aggregate(model)).message.content).toEqual([new TextBlock('Recovered')])
    },
  )

  it('closes a started native tool block when cancelled', async () => {
    const { model, create, stop } = setup()
    const hold = pause()
    const waiting = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({
            tool_calls: [
              { index: 0, id: 'call-1', function: { name: 'weather', arguments: '{"city":' } },
            ],
          })
          waiting.resume()
          await hold.wait
        } finally {
          hold.resume()
        }
      })(),
    )
    stop.mockImplementation(async () => hold.resume())

    const controller = new AbortController()
    const stream = model.stream(messages, { toolSpecs: tools, cancelSignal: controller.signal })
    try {
      await stream.next()
      await stream.next()
      await stream.next()
      const rest = collect(stream)
      await waiting.wait
      controller.abort()
      const events = await rest

      expect(events).toContainEqual({ type: 'modelContentBlockStopEvent' })
      expect(events.at(-1)).toEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
    } finally {
      hold.resume()
      await stream.return(undefined)
    }
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
})
