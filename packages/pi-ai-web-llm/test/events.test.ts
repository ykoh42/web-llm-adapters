import { runInNewContext } from 'node:vm'
import { describe, expect, it } from 'vitest'
import type { StreamChunk } from '../../../shared/web-llm/stream/chunks.js'
import { responseStream } from '../src/events.js'
import { collect, setup } from './helpers.js'

function chunk(values: Partial<StreamChunk>): StreamChunk {
  return { text: '', reasoning: '', calls: [], stop: undefined, usage: undefined, ...values }
}

function stream(...values: Partial<StreamChunk>[]) {
  const { model } = setup()
  return responseStream(model, () => ({
    chunks: (async function* () {
      for (const value of values) yield chunk(value)
    })(),
    toolPolicy: { choice: undefined, toolNames: ['lookup'] },
  }))
}

const call = (index: number, input: string, id = `call-${index}`) => ({
  index,
  id,
  name: 'lookup',
  input,
})

describe('Pi response events', () => {
  it('preserves the message of an error-like rejection', async () => {
    const { model } = setup()
    const response = responseStream(model, () => {
      throw { message: 'GPU unavailable' }
    })

    expect(await response.result()).toMatchObject({
      stopReason: 'error',
      errorMessage: 'GPU unavailable',
    })
  })

  it('marks AbortError failures from another realm as aborted', async () => {
    const { model } = setup()
    const response = responseStream(model, () => ({
      chunks: (async function* () {
        yield chunk({})
        throw runInNewContext(
          'Object.assign(new Error("Generation interrupted."), { name: "AbortError" })',
        )
      })(),
      toolPolicy: { choice: undefined, toolNames: ['lookup'] },
    }))

    expect(await response.result()).toMatchObject({
      stopReason: 'aborted',
      errorMessage: 'Generation interrupted.',
    })
  })

  it.each([
    { name: 'null prototype', error: () => Object.create(null) },
    {
      name: 'throwing string conversion',
      error: () => ({
        toString() {
          throw new Error('Cannot convert this error')
        },
      }),
    },
    {
      name: 'throwing message getter',
      error: () =>
        Object.defineProperty(new Error(), 'message', {
          get() {
            throw new Error('Cannot read this error')
          },
        }),
    },
  ])('finishes with an error event for $name failures', async ({ error }) => {
    const { model } = setup()
    const response = responseStream(model, () => {
      throw error()
    })

    expect((await collect(response)).map((event) => event.type)).toEqual(['error'])
    expect(await response.result()).toMatchObject({
      stopReason: 'error',
      errorMessage: 'The request failed with an unreadable error.',
      content: [],
    })
  })

  it('closes each text or thinking block before opening the next one', async () => {
    const response = stream(
      { reasoning: 'Plan' },
      { reasoning: ' carefully' },
      { text: 'First' },
      { reasoning: 'Reconsider' },
      { text: 'Final', stop: 'endTurn' },
    )
    const events = await collect(response)
    expect(events.map((event) => event.type)).toEqual([
      'start',
      'thinking_start',
      'thinking_delta',
      'thinking_delta',
      'thinking_end',
      'text_start',
      'text_delta',
      'text_end',
      'thinking_start',
      'thinking_delta',
      'thinking_end',
      'text_start',
      'text_delta',
      'text_end',
      'done',
    ])
    expect(
      events.filter((event) => event.type === 'thinking_end' || event.type === 'text_end'),
    ).toEqual([
      expect.objectContaining({ contentIndex: 0, content: 'Plan carefully' }),
      expect.objectContaining({ contentIndex: 1, content: 'First' }),
      expect.objectContaining({ contentIndex: 2, content: 'Reconsider' }),
      expect.objectContaining({ contentIndex: 3, content: 'Final' }),
    ])
    expect((await response.result()).content).toEqual([
      { type: 'thinking', thinking: 'Plan carefully' },
      { type: 'text', text: 'First' },
      { type: 'thinking', thinking: 'Reconsider' },
      { type: 'text', text: 'Final' },
    ])
  })

  it('publishes calls by index with their own original argument bytes', async () => {
    const response = stream({
      calls: [call(3, '{ "code": "B8" }'), call(1, '{"code":"A7"}')],
      stop: 'toolUse',
    })
    const events = await collect(response)
    expect(events.filter((event) => event.type === 'toolcall_delta')).toEqual([
      expect.objectContaining({ contentIndex: 0, delta: '{"code":"A7"}' }),
      expect.objectContaining({ contentIndex: 1, delta: '{ "code": "B8" }' }),
    ])
    expect((await response.result()).content).toEqual([
      { type: 'toolCall', id: 'call-1', name: 'lookup', arguments: { code: 'A7' } },
      { type: 'toolCall', id: 'call-3', name: 'lookup', arguments: { code: 'B8' } },
    ])
  })

  it('streams native tool argument fragments in output order', async () => {
    const response = stream(
      { calls: [call(0, '{"code":"A')] },
      { text: 'Between' },
      { calls: [{ index: 0, id: undefined, name: undefined, input: '7"}' }] },
      { stop: 'toolUse' },
    )
    const events = await collect(response)

    expect(events.map((event) => event.type)).toEqual([
      'start',
      'toolcall_start',
      'toolcall_delta',
      'text_start',
      'text_delta',
      'toolcall_delta',
      'text_end',
      'toolcall_end',
      'done',
    ])
    expect(
      events.filter((event) => event.type === 'toolcall_delta').map((event) => event.delta),
    ).toEqual(['{"code":"A', '7"}'])
    expect((await response.result()).content).toEqual([
      { type: 'toolCall', id: 'call-0', name: 'lookup', arguments: { code: 'A7' } },
      { type: 'text', text: 'Between' },
    ])
  })

  it('keeps partial arguments available while WebLLM is still generating', async () => {
    const { model } = setup()
    let resume!: () => void
    let finish!: () => void
    let reachedPause!: () => void
    let reachedSecondPause!: () => void
    const gate = new Promise<void>((resolve) => (resume = resolve))
    const finishGate = new Promise<void>((resolve) => (finish = resolve))
    const paused = new Promise<void>((resolve) => (reachedPause = resolve))
    const secondPause = new Promise<void>((resolve) => (reachedSecondPause = resolve))
    const response = responseStream(model, () => ({
      chunks: (async function* () {
        yield chunk({ calls: [call(0, '{"code":"A')] })
        reachedPause()
        await gate
        yield chunk({ calls: [{ index: 0, id: undefined, name: undefined, input: '7"}' }] })
        reachedSecondPause()
        await finishGate
        yield chunk({ stop: 'toolUse' })
      })(),
      toolPolicy: { choice: undefined, toolNames: ['lookup'] },
    }))
    const iterator = response[Symbol.asyncIterator]()

    try {
      expect((await iterator.next()).value?.type).toBe('start')
      const start = await iterator.next()
      const delta = await iterator.next()
      await paused

      expect(start.value?.type).toBe('toolcall_start')
      expect(delta.value?.type).toBe('toolcall_delta')
      if (delta.value?.type === 'toolcall_delta')
        expect(delta.value.partial.content[0]).toMatchObject({
          type: 'toolCall',
          arguments: { code: 'A' },
        })

      resume()
      const nextDelta = await iterator.next()
      await secondPause
      expect(nextDelta.value?.type).toBe('toolcall_delta')
      if (nextDelta.value?.type === 'toolcall_delta')
        expect(nextDelta.value.partial.content[0]).toMatchObject({
          type: 'toolCall',
          arguments: { code: 'A7' },
        })
      finish()
      let next = await iterator.next()
      while (!next.done) next = await iterator.next()
      expect((await response.result()).content[0]).toMatchObject({
        type: 'toolCall',
        arguments: { code: 'A7' },
      })
    } finally {
      resume()
      finish()
    }
  })

  it.each([
    [call(0, '{}'), call(1, '{')],
    [call(0, '{}', 'same-id'), call(1, '{}', 'same-id')],
  ])('withholds all executable calls when any call is invalid %#', async (...calls) => {
    const response = stream({ calls, stop: 'toolUse' })
    const events = await collect(response)
    expect(events.at(-1)?.type).toBe('error')
    expect(events.filter((event) => event.type === 'toolcall_end')).toEqual([])
    expect(await response.result()).toMatchObject({
      stopReason: 'error',
      content: [
        { type: 'toolCall', name: 'lookup' },
        { type: 'toolCall', name: 'lookup' },
      ],
    })
  })

  it('does not complete a tool call truncated by the token limit', async () => {
    const response = stream({ calls: [call(0, '{"code":"A')], stop: 'maxTokens' })
    const events = await collect(response)

    expect(events.map((event) => event.type)).toEqual([
      'start',
      'toolcall_start',
      'toolcall_delta',
      'done',
    ])
    expect(await response.result()).toMatchObject({ stopReason: 'length' })
  })

  it('keeps out-of-order tool fragments when generation reaches the token limit', async () => {
    const response = stream(
      { calls: [call(2, '{"code":"B'), call(1, '{"code":"A')] },
      { calls: [{ index: 2, id: undefined, name: undefined, input: '8' }], stop: 'maxTokens' },
    )
    const events = await collect(response)

    expect(events.filter((event) => event.type === 'toolcall_delta')).toMatchObject([
      { delta: '{"code":"A' },
      { delta: '{"code":"B8' },
    ])
    expect(events.filter((event) => event.type === 'toolcall_end')).toEqual([])
    expect(await response.result()).toMatchObject({
      stopReason: 'length',
      content: [
        { type: 'toolCall', id: 'call-1', arguments: { code: 'A' } },
        { type: 'toolCall', id: 'call-2', arguments: { code: 'B8' } },
      ],
    })
  })

  it('keeps final usage received after the stop chunk', async () => {
    const response = stream(
      { text: 'Done', stop: 'endTurn' },
      {
        usage: {
          prompt_tokens: 5,
          completion_tokens: 2,
          total_tokens: 7,
          extra: {
            e2e_latency_s: 0,
            prefill_tokens_per_s: 0,
            decode_tokens_per_s: 0,
            time_to_first_token_s: 0,
            time_per_output_token_s: 0,
          },
        },
      },
    )
    expect(await response.result()).toMatchObject({
      stopReason: 'stop',
      content: [{ type: 'text', text: 'Done' }],
      usage: { input: 5, output: 2, totalTokens: 7 },
    })
  })
})
