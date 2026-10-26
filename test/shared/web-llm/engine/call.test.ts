import { expect, it, vi } from 'vitest'
import { startCall, streamCall } from '../../../../shared/web-llm/engine/call.js'
import { setupEngine, chunks, chunk, pause, collect } from '../../../../fixtures/web-llm.js'

it.each(['throw', 'reject'] as const)(
  'closes the stream and releases the queue when interrupt fails (%s)',
  async (failure) => {
    const { engine, stop } = setupEngine()
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
  const { engine, stop } = setupEngine()
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

it('keeps the original stream failure when cleanup also fails', async () => {
  const { engine, stop } = setupEngine()
  const failure = new Error('Read failed')
  stop.mockRejectedValueOnce(new Error('Cleanup failed'))
  let returned = false
  const source = (async function* () {
    try {
      yield 1
      yield 2
    } finally {
      returned = true
    }
  })()
  const stream = streamCall(
    engine,
    async () => source,
    async function* (chunks) {
      for await (const value of chunks) {
        yield value
        throw failure
      }
    },
  )
  await expect(collect(stream)).rejects.toBe(failure)
  expect(returned).toBe(true)
  const next = await startCall(engine, async () => chunks(chunk({}, 'stop')))
  await next.close(false)
})

it('does not open a request cancelled while registering its active listener', async () => {
  const { engine } = setupEngine()
  const controller = new AbortController()
  const listen = controller.signal.addEventListener.bind(controller.signal)
  let registrations = 0
  vi.spyOn(controller.signal, 'addEventListener').mockImplementation((...args) => {
    listen(...args)
    if (++registrations === 2) controller.abort(new Error('Cancelled before opening'))
  })
  const open = vi.fn(async () => chunks(chunk({}, 'stop')))
  await expect(startCall(engine, open, controller.signal)).rejects.toThrow(
    'Cancelled before opening',
  )
  expect(open).not.toHaveBeenCalled()
  const next = await startCall(engine, async () => chunks(chunk({}, 'stop')))
  await next.close(false)
})
