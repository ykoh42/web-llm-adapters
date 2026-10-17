import { expect, it, vi } from 'vitest'
import { startCall } from '../../src/engine/call.js'
import { setup, chunks, chunk, pause } from '../../fixtures/engine.js'

it.each(['throw', 'reject'] as const)(
  'closes the stream and releases the queue when interrupt fails (%s)',
  async (failure) => {
    const { engine, stop } = setup()
    stop.mockImplementation(() => {
      if (failure === 'throw') throw new Error('Interrupt failed')
      return Promise.reject(new Error('Interrupt failed'))
    })
    const returned = vi.fn(async () => ({ done: true as const, value: undefined }))
    const source = {
      [Symbol.asyncIterator]: () => ({
        next: async () => ({ done: false as const, value: 1 }),
        return: returned,
      }),
    }
    const controller = new AbortController()
    const first = await startCall(engine, async () => source, controller.signal)
    controller.abort()
    await expect(first.close(true)).rejects.toThrow('Interrupt failed')
    expect(returned).toHaveBeenCalledTimes(1)
    const second = await startCall(engine, async () => chunks(chunk({}, 'stop')))
    await second.close(false)
  },
)

it('starts stream cleanup while an async interrupt is still pending', async () => {
  const { engine, stop } = setup()
  const closing = pause()
  stop.mockImplementation(() => closing.wait)
  const returned = vi.fn(async () => {
    closing.resume()
    return { done: true as const, value: undefined }
  })
  const call = await startCall(engine, async () => ({
    [Symbol.asyncIterator]: () => ({
      next: async () => ({ done: true as const, value: undefined }),
      return: returned,
    }),
  }))
  await call.close(true)
  expect(returned).toHaveBeenCalledTimes(1)
})
