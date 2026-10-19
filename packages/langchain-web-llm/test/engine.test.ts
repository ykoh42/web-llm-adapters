import { describe, expect, it } from 'vitest'
import { ChatWebLLM, withEngine } from '../src/index.js'
import { withEngine as withStrandsEngine } from '../../strands-web-llm/src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

describe('engine ownership', () => {
  it('serializes concurrent LangChain calls on one engine', async () => {
    const { model, create } = setup()
    const hold = pause()
    let active = 0
    let peak = 0
    create.mockImplementation(async () =>
      (async function* () {
        active++
        peak = Math.max(peak, active)
        try {
          yield chunk({ content: 'A' })
          await hold.wait
          yield chunk({ content: 'B' }, 'stop')
        } finally {
          active--
        }
      })(),
    )
    const first = model.invoke('One')
    const second = model.invoke('Two')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(create).toHaveBeenCalledTimes(1)
    hold.resume()
    expect((await Promise.all([first, second])).map((message) => message.text)).toEqual([
      'AB',
      'AB',
    ])
    expect(peak).toBe(1)
  })

  it('shares the native queue across Strands and LangChain adapters', async () => {
    const { engine, model, create } = setup(chunk({ content: 'Okay' }, 'stop'))
    const hold = pause()
    const native = withStrandsEngine(engine, async () => {
      await hold.wait
    })
    const result = model.invoke('Hi')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(create).not.toHaveBeenCalled()
    hold.resume()
    await native
    expect((await result).text).toBe('Okay')
  })

  it('cancels a queued call without interrupting the active call', async () => {
    const { engine, model, stop, create } = setup(chunk({ content: 'Okay' }, 'stop'))
    const hold = pause()
    const native = withEngine(engine, async () => {
      await hold.wait
    })
    const cancel = new AbortController()
    const failed = expect(
      model.invoke('Cancelled', { signal: cancel.signal }),
    ).rejects.toBeDefined()
    cancel.abort(new Error('Stop waiting'))
    await failed
    expect(stop).not.toHaveBeenCalled()
    hold.resume()
    await native
    expect((await model.invoke('After')).text).toBe('Okay')
    expect(create).toHaveBeenCalledTimes(1)
  })

  it('interrupts active generation and allows the next call to finish', async () => {
    const { model, create, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'Started' })
        await hold.wait
        yield chunk({}, 'abort')
      })(),
    )
    stop.mockImplementation(async () => {
      hold.resume()
    })
    const cancel = new AbortController()
    const stream = await model.stream('Hi', { signal: cancel.signal })
    expect((await stream.next()).value?.text).toBe('Started')
    cancel.abort()
    await expect(collect(stream)).rejects.toBeDefined()
    expect((await model.invoke('After')).text).toBe('Recovered')
    expect(stop).toHaveBeenCalledTimes(1)
  })

  it('closes a stream when a consumer stops reading', async () => {
    const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
    let closed = false
    create.mockResolvedValueOnce(
      (async function* () {
        try {
          yield chunk({ content: 'Started' })
          yield chunk({}, 'stop')
        } finally {
          closed = true
        }
      })(),
    )
    for await (const item of await model.stream('Hi')) {
      expect(item.text).toBe('Started')
      break
    }
    expect(closed).toBe(true)
    expect((await model.invoke('After')).text).toBe('Recovered')
  })

  it('releases the engine after an opening failure or a chunk callback failure', async () => {
    const { engine, create } = setup(chunk({ content: 'Okay' }, 'stop'))
    create.mockRejectedValueOnce(new Error('Native failure'))
    const model = new ChatWebLLM({ engine, modelId: 'test-model' })
    await expect(model.invoke('Hi')).rejects.toThrow('Native failure')
    const failed = new ChatWebLLM({
      engine,
      modelId: 'test-model',
      onChunk: () => {
        throw new Error('Callback failure')
      },
    })
    await expect(failed.invoke('Hi')).rejects.toThrow('Callback failure')
    expect((await model.invoke('After')).text).toBe('Okay')
  })

  it('passes a native context error through without rewriting it', async () => {
    const { model, create } = setup()
    const error = new Error('Context full')
    error.name = 'ContextWindowSizeExceededError'
    create.mockRejectedValueOnce(error)
    await expect(model.invoke('Hi')).rejects.toBe(error)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Okay' }, 'stop')))
    expect((await model.invoke('After')).text).toBe('Okay')
  })
})

it('honors explicit opening retries without retrying a partial response', async () => {
  const { engine, create } = setup(chunk({ content: 'Okay' }, 'stop'))
  create.mockRejectedValueOnce(new Error('Opening failed'))
  const model = new ChatWebLLM({ engine, modelId: 'test-model', maxRetries: 1 })
  expect((await model.invoke('Hi')).text).toBe('Okay')
  expect(create).toHaveBeenCalledTimes(2)
  create.mockResolvedValueOnce(
    (async function* () {
      yield chunk({ content: 'Partial' })
      throw new Error('Stream failed')
    })(),
  )
  await expect(model.invoke('Hi')).rejects.toThrow('Stream failed')
  expect(create).toHaveBeenCalledTimes(3)
})

it('shares native scheduling between the independently bundled npm packages', async () => {
  const { withEngine: useStrands } = await import('../../strands-web-llm/dist/index.js')
  const { withEngine: useLangChain } = await import('../dist/index.js')
  const { engine } = setup()
  const hold = pause()
  let entered = false
  const first = useStrands(engine, async () => {
    await hold.wait
  })
  const second = useLangChain(engine, async () => {
    entered = true
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(entered).toBe(false)
  hold.resume()
  await Promise.all([first, second])
  expect(entered).toBe(true)
})
