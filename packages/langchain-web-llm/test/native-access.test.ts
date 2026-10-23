import { expect, it, vi } from 'vitest'
import { setImmediate } from 'node:timers/promises'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { createBrowserBridge } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

it('runs browser native work between calls sharing the same engine', async () => {
  const { engine, model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
  const bridge = createBrowserBridge(engine)
  const hold = pause()
  create.mockResolvedValueOnce(
    (async function* () {
      yield chunk({ content: 'Started' })
      await hold.wait
      yield chunk({}, 'stop')
    })(),
  )
  const stream = await model.stream('First')
  const run = vi.fn((native: MLCEngineInterface) => {
    expect(native).toBe(engine)
    expect(create).toHaveBeenCalledTimes(1)
    return 42
  })
  try {
    expect((await stream.next()).value?.text).toBe('Started')
    const native = bridge.withEngine(run)
    const next = model.invoke('Next')
    await Promise.resolve()
    expect(run).not.toHaveBeenCalled()
    hold.resume()
    await collect(stream)
    expect(await native).toBe(42)
    expect((await next).text).toBe('Recovered')
    expect(run).toHaveBeenCalledOnce()
  } finally {
    hold.resume()
    await stream.return?.()
    await bridge.close()
  }
})

it('waits for native work at close and rejects queued and later callbacks', async () => {
  const { engine } = setup()
  const bridge = createBrowserBridge(engine)
  const entered = pause()
  const hold = pause()
  const unload = vi.spyOn(engine, 'unload')
  const queued = vi.fn()
  let closed = false
  try {
    const native = bridge.withEngine(async () => {
      entered.resume()
      await hold.wait
      return 42
    })
    await entered.wait
    const rejected = expect(bridge.withEngine(queued)).rejects.toThrow('closed')
    const closing = bridge.close().then(() => {
      closed = true
    })
    await setImmediate()
    expect(closed).toBe(false)
    expect(queued).not.toHaveBeenCalled()
    hold.resume()
    expect(await native).toBe(42)
    await closing
    await rejected
    await expect(bridge.withEngine(queued)).rejects.toThrow('closed')
    expect(queued).not.toHaveBeenCalled()
    expect(unload).not.toHaveBeenCalled()
  } finally {
    hold.resume()
    await bridge.close()
  }
})

it('preserves native callback failures and releases the engine', async () => {
  const { engine, model } = setup(chunk({ content: 'Recovered' }, 'stop'))
  const bridge = createBrowserBridge(engine)
  const failure = new TypeError('Native work failed')
  try {
    await expect(
      bridge.withEngine(async () => {
        throw failure
      }),
    ).rejects.toBe(failure)
    expect((await model.invoke('Next')).text).toBe('Recovered')
  } finally {
    await bridge.close()
  }
})

it('closes escaped native streams before releasing the engine', async () => {
  const { engine, model, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
  const bridge = createBrowserBridge(engine)
  const stream = chunks(chunk({ content: 'Native' }, 'stop'))
  const finish = vi.spyOn(stream, 'return')
  try {
    await expect(bridge.withEngine(() => stream)).rejects.toThrow(
      'Consume native streams inside withEngine().',
    )
    expect(finish).toHaveBeenCalledOnce()
    expect(stop).toHaveBeenCalledOnce()
    expect((await model.invoke('Next')).text).toBe('Recovered')
  } finally {
    await bridge.close()
  }
})
