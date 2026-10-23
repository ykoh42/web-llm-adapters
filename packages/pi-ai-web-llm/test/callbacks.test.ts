import { describe, expect, it, vi } from 'vitest'
import { setup } from './helpers.js'

const context = { messages: [{ role: 'user' as const, content: 'Hello', timestamp: 0 }] }

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
