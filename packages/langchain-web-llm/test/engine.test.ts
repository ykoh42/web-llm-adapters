import { describe, expect, it, vi } from 'vitest'
import { ContextOverflowError } from '@langchain/core/errors'
import { HumanMessage } from '@langchain/core/messages'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  type ChatCompletionRequest,
} from '@mlc-ai/web-llm'
import { ChatWebLLM, withEngine } from '../src/index.js'
import { withEngine as withStrandsEngine } from '../../strands-web-llm/src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

describe('engine ownership', () => {
  it.each([
    [0, 1],
    [1, 0],
  ])('overrides %i model retries with %i call retries', async (defaults, retries) => {
    const { engine, create } = setup(chunk({ content: 'Okay' }, 'stop'))
    create.mockRejectedValueOnce(new Error('Opening failed'))
    const model = new ChatWebLLM({ engine, modelId: 'test-model', maxRetries: defaults })

    const result = model.invoke('Hi', { maxRetries: retries })
    if (retries) expect((await result).text).toBe('Okay')
    else await expect(result).rejects.toThrow('Opening failed')
    expect(create).toHaveBeenCalledTimes(retries + 1)
  })

  it('retries a Hermes request after WebLLM mutates its first attempt', async () => {
    const { engine, create } = setup()
    const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
    const attempts: ChatCompletionRequest[] = []
    create.mockImplementation(async (request) => {
      attempts.push(structuredClone(request))
      postInitAndCheckFieldsChatCompletion(request, modelId, ModelType.LLM)
      if (attempts.length === 1) throw new Error('Opening failed')
      return chunks(
        chunk(
          { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: '{"code":"A7"}' } }] },
          'tool_calls',
        ),
      )
    })
    const model = new ChatWebLLM({ engine, modelId, maxRetries: 1 })
    const result = await model
      .bindTools([
        {
          type: 'function',
          function: {
            name: 'lookup',
            parameters: { type: 'object', properties: { code: { type: 'string' } } },
          },
        },
      ])
      .invoke('Look up A7.')

    expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expect(attempts).toHaveLength(2)
    expect(attempts[1]).toEqual(attempts[0])
    expect(attempts[1]?.response_format).toBeUndefined()
    expect(attempts[1]?.messages).toEqual([{ role: 'user', content: 'Look up A7.' }])
  })

  it('keeps local retries off when maxRetries is undefined', async () => {
    const { engine, create } = setup()
    create.mockRejectedValue(new Error('GPU lost'))
    const options = { engine, modelId: 'test-model' }
    Object.assign(options, { maxRetries: undefined })
    const model = new ChatWebLLM(options)

    await expect(model.invoke('Hi')).rejects.toThrow('GPU lost')
    expect(create).toHaveBeenCalledOnce()
  })

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

  it('keeps the engine until a cancelled opening request settles', async () => {
    const { model, create, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const opening = pause()
    const hold = pause()
    const iterator = {
      next: vi.fn(async () => ({ done: true as const, value: undefined })),
      return: vi.fn(async () => ({ done: true as const, value: undefined })),
    }
    create.mockImplementationOnce(async () => {
      opening.resume()
      await hold.wait
      return { [Symbol.asyncIterator]: () => iterator }
    })
    const cancel = new AbortController()
    const first = model.invoke('First', { signal: cancel.signal })
    await opening.wait
    cancel.abort()
    const failed = expect(first).rejects.toBeDefined()
    const second = model.invoke('Second')

    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(create).toHaveBeenCalledTimes(1)

    hold.resume()
    await failed
    expect((await second).text).toBe('Recovered')
    expect(stop).toHaveBeenCalledTimes(1)
    expect(iterator.return).toHaveBeenCalledOnce()
  })

  it('preserves cancellation when closing a late opening stream also fails', async () => {
    const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const opening = pause()
    const hold = pause()
    const returned = vi.fn(async () => {
      throw new Error('Cleanup failed')
    })
    create.mockImplementationOnce(async () => {
      opening.resume()
      await hold.wait
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: true as const, value: undefined }),
          return: returned,
        }),
      }
    })
    const cancel = new AbortController()
    const first = collect(
      model._streamResponseChunks([new HumanMessage('First')], { signal: cancel.signal }),
    )
    await opening.wait
    cancel.abort()
    const failed = expect(first).rejects.toHaveProperty('name', 'AbortError')
    const second = model.invoke('Second')
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(create).toHaveBeenCalledTimes(1)

    hold.resume()
    await failed
    expect(returned).toHaveBeenCalledOnce()
    expect((await second).text).toBe('Recovered')
  })

  it('cancels retry backoff without delaying the next engine call', async () => {
    vi.useFakeTimers()
    try {
      const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
      const failed = pause()
      create.mockImplementationOnce(async () => {
        failed.resume()
        throw new Error('Opening failed')
      })
      const model = new ChatWebLLM({
        engine,
        modelId: 'test-model',
        maxConcurrency: 1,
        maxRetries: 1,
      })
      const cancel = new AbortController()
      const first = model.invoke('First', { signal: cancel.signal })

      await failed.wait
      await vi.advanceTimersByTimeAsync(0)
      expect(vi.getTimerCount()).toBeGreaterThan(0)

      cancel.abort()
      await expect(first).rejects.toBeDefined()
      expect((await model.invoke('Second')).text).toBe('Recovered')
      await vi.runOnlyPendingTimersAsync()
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
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

  it('preserves a native context error as the cause of the standard error', async () => {
    const { model, create } = setup()
    const error = new Error('Context full')
    error.name = 'ContextWindowSizeExceededError'
    create.mockRejectedValueOnce(error)
    const failed = model.invoke('Hi')
    await expect(failed).rejects.toBeInstanceOf(ContextOverflowError)
    await expect(failed).rejects.toHaveProperty('cause', error)
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
