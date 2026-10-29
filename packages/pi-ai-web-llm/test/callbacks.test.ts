import { describe, expect, it, vi } from 'vitest'
import { getEventListeners } from 'node:events'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import { chunk, setup } from './helpers.js'

const context = { messages: [{ role: 'user' as const, content: 'Hello', timestamp: 0 }] }

it('does not let the provider stream observer change parsed output', async () => {
  const native = chunk({ content: 'Original' }, 'stop')
  native.choices[0]!.logprobs = {
    content: [{ token: 'Original', logprob: -0.1, bytes: null, top_logprobs: [] }],
  }
  const { models, model } = setup([native])
  const observed: unknown[] = []
  const response = await models.completeSimple(model, context, {
    onProviderStreamEvent: (event) => {
      observed.push(event)
      if (isChatChunk(event)) event.choices[0]!.delta.content = 'Changed'
    },
  })

  expect(response.content).toEqual([{ type: 'text', text: 'Original' }])
  expect(observed).toHaveLength(1)
  expect(isChatChunk(observed[0])).toBe(true)
  if (isChatChunk(observed[0]))
    expect(observed[0]).toMatchObject({
      id: 'response',
      model: 'loaded-model',
      created: 0,
      choices: [{ logprobs: { content: [{ token: 'Original', logprob: -0.1 }] } }],
    })
})

it('awaits async provider chunk callbacks and reports their failures', async () => {
  const { models, model, create } = setup(undefined, {
    onChunk: async () => {
      await Promise.resolve()
      throw new Error('Async observer failed')
    },
  })
  const response = await models.completeSimple(model, context)

  expect(response).toMatchObject({
    stopReason: 'error',
    errorMessage: 'Async observer failed',
    content: [],
  })
  expect(create).toHaveBeenCalledOnce()
})

it('keeps parsed output when an async provider chunk callback mutates its copy', async () => {
  const native = chunk({ content: 'Original' }, 'stop')
  const observed: unknown[] = []
  const { models, model } = setup([native], {
    onChunk: async (value) => {
      await Promise.resolve()
      observed.push(value)
      if (isChatChunk(value)) value.choices[0]!.delta.content = 'Changed'
    },
  })
  const response = await models.completeSimple(model, context)

  expect(response.content).toEqual([{ type: 'text', text: 'Original' }])
  expect(observed).toHaveLength(1)
})

it.each([
  { timeout: false, missing: false },
  { timeout: true, missing: false },
  { timeout: true, missing: true },
])(
  'cancels an active callback (timeout: $timeout, missing any: $missing)',
  async ({ timeout, missing }) => {
    const original = Object.getOwnPropertyDescriptor(AbortSignal, 'any')!
    if (missing) Object.defineProperty(AbortSignal, 'any', { ...original, value: undefined })
    let finish!: () => void
    const hold = new Promise<void>((resolve) => {
      finish = resolve
    })
    const onChunk = vi.fn().mockImplementationOnce(() => hold)
    const { models, model, create } = setup(undefined, { onChunk })
    const close = vi.fn(async () => ({ done: true as const, value: undefined }))
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]: () => ({
        next: async () => ({ done: false as const, value: chunk({ content: 'Hello' }) }),
        return: close,
      }),
    })
    const controller = new AbortController()
    const pending = models.completeSimple(model, context, {
      signal: controller.signal,
      ...(timeout && { timeoutMs: 5000 }),
    })
    try {
      await vi.waitFor(() => expect(onChunk).toHaveBeenCalledOnce())
      controller.abort()
      expect((await pending).stopReason).toBe('aborted')
      expect(create).toHaveBeenCalledOnce()
      await vi.waitFor(() => expect(close).toHaveBeenCalledOnce())
      expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
      expect(onChunk).toHaveBeenCalledTimes(2)
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      finish()
      if (missing) Object.defineProperty(AbortSignal, 'any', original)
    }
  },
)

it('times out without AbortSignal.any when there is no caller signal', async () => {
  const { models, model, create } = setup()
  const any = vi.spyOn(AbortSignal, 'any').mockImplementation(() => {
    throw new TypeError('AbortSignal.any is unavailable')
  })
  try {
    const response = await models.completeSimple(model, context, {
      timeoutMs: 100,
      onPayload: () => new Promise<void>(() => {}),
    })
    expect(response.stopReason).toBe('aborted')
    expect(create).not.toHaveBeenCalled()
  } finally {
    any.mockRestore()
  }
})

it.each(['caller', 'timeout'] as const)(
  'honors %s cancellation without AbortSignal.any',
  async (source) => {
    const original = Object.getOwnPropertyDescriptor(AbortSignal, 'any')!
    Object.defineProperty(AbortSignal, 'any', { ...original, value: undefined })
    try {
      const { models, model, create } = setup()
      const controller = new AbortController()
      const onPayload = vi.fn(() => new Promise<void>(() => {}))
      const pending = models.completeSimple(model, context, {
        signal: controller.signal,
        timeoutMs: source === 'timeout' ? 100 : 5000,
        onPayload,
      })
      if (source === 'caller') {
        await vi.waitFor(() => expect(onPayload).toHaveBeenCalledOnce())
        controller.abort()
      }
      expect((await pending).stopReason).toBe('aborted')
      expect(create).not.toHaveBeenCalled()
      expect(getEventListeners(controller.signal, 'abort')).toHaveLength(0)
    } finally {
      Object.defineProperty(AbortSignal, 'any', original)
    }
  },
)

it('removes combined-signal listeners after a successful call without AbortSignal.any', async () => {
  const original = Object.getOwnPropertyDescriptor(AbortSignal, 'any')!
  Object.defineProperty(AbortSignal, 'any', { ...original, value: undefined })
  const timeout = new AbortController()
  const timer = vi.spyOn(AbortSignal, 'timeout').mockReturnValue(timeout.signal)
  try {
    const { models, model } = setup()
    const controller = new AbortController()
    const result = await models.completeSimple(model, context, {
      signal: controller.signal,
      timeoutMs: 5000,
    })
    expect(result.stopReason).toBe('stop')
    expect(getEventListeners(controller.signal, 'abort')).toHaveLength(0)
    expect(getEventListeners(timeout.signal, 'abort')).toHaveLength(0)
  } finally {
    timer.mockRestore()
    Object.defineProperty(AbortSignal, 'any', original)
  }
})

it.each(['onPayload', 'onProviderStreamEvent'] as const)(
  '%s reports an unreadable failure and releases the engine',
  async (callback) => {
    const { models, model, create } = setup()
    const response = await models.completeSimple(model, context, {
      [callback]: () => {
        throw Object.create(null)
      },
    })

    expect(response).toMatchObject({
      stopReason: 'error',
      errorMessage: 'The request failed with an unreadable error.',
      content: [],
    })
    expect(create).toHaveBeenCalledTimes(callback === 'onPayload' ? 0 : 1)
    expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
    expect(create).toHaveBeenCalledTimes(callback === 'onPayload' ? 1 : 2)
  },
)

describe.each(['onPayload', 'onProviderStreamEvent'] as const)('%s cancellation', (callback) => {
  it.each([
    { mode: 'abort', late: 'resolve' },
    { mode: 'abort', late: 'reject' },
    { mode: 'timeout', late: 'resolve' },
    { mode: 'timeout', late: 'reject' },
  ])('finishes on $mode before a late $late and releases the engine', async ({ mode, late }) => {
    const { models, model, create } = setup()
    const controller = new AbortController()
    let finish!: () => void
    let fail!: (error: Error) => void
    const hold = new Promise<void>((resolve, reject) => {
      finish = resolve
      fail = reject
    })
    const handler = vi.fn(() => hold)
    const pending = models.completeSimple(model, context, {
      [callback]: handler,
      signal: controller.signal,
      ...(mode === 'timeout' && { timeoutMs: 100 }),
    })
    try {
      await vi.waitFor(() => expect(handler).toHaveBeenCalledOnce())
      if (mode === 'abort') controller.abort()
      expect((await pending).stopReason).toBe('aborted')
      expect(create).toHaveBeenCalledTimes(callback === 'onPayload' ? 0 : 1)
      expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
    } finally {
      if (late === 'resolve') finish()
      else fail(new Error('Late callback failure'))
    }
    await Promise.resolve()
    expect((await pending).stopReason).toBe('aborted')
    expect(create).toHaveBeenCalledTimes(callback === 'onPayload' ? 1 : 2)
  })
})

function isChatChunk(event: unknown): event is ChatCompletionChunk {
  return (
    typeof event === 'object' &&
    event !== null &&
    'object' in event &&
    event.object === 'chat.completion.chunk'
  )
}
