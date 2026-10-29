import { afterEach, describe, expect, it, vi } from 'vitest'
import { getRetryable, stampRetryable } from '@langchain/core/errors'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserBridge,
  type BrowserPage,
  type ChatWebLLMOptions,
} from '../src/index.js'
import { chunk, chunks, finishRetries, pause, setup } from '../fixtures/engine.js'
import { SignalCaller } from '../src/chat/caller.js'

const bridges: BrowserBridge[] = []
function browser(options: Omit<ChatWebLLMOptions, 'modelId' | 'engine'> = {}) {
  const result = setup(chunk({ content: 'Answer' }, 'stop'))
  const bridge = createBrowserBridge(result.engine)
  bridges.push(bridge)
  vi.stubGlobal('window', { webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  const model = new BrowserChatWebLLM(page, { modelId: 'test-model', ...options })
  return { ...result, page, model, bridge }
}

afterEach(async () => {
  try {
    await Promise.all(bridges.splice(0).map((bridge) => bridge.close()))
  } finally {
    vi.unstubAllGlobals()
  }
})

it.each([
  ...(['local', 'browser'] as const).flatMap((mode) => [
    { mode, error: { message: 'Invalid request', status: 400 } },
    { mode, error: { message: 'Invalid request', statusCode: 400 } },
    { mode, error: { message: 'Invalid request', response: { status: 400 } } },
    { mode, error: stampRetryable({ message: 'Do not retry' }, false) },
  ]),
])('$mode respects retry information in an error-like object', async ({ mode, error }) => {
  const fixture = browser({ maxRetries: 1 })
  fixture.create.mockRejectedValue(error)
  const model =
    mode === 'local'
      ? new ChatWebLLM({ engine: fixture.engine, modelId: 'test-model', maxRetries: 1 })
      : fixture.model

  await expect(model.invoke('Hi')).rejects.toMatchObject(error)
  expect(fixture.create).toHaveBeenCalledOnce()
})

it.each(['local', 'browser'] as const)(
  '%s does not retry an error-like cancellation',
  async (mode) => {
    const fixture = browser({ maxRetries: 1 })
    fixture.create.mockRejectedValueOnce({ name: 'AbortError', message: 'Generation cancelled.' })
    const model =
      mode === 'local'
        ? new ChatWebLLM({ engine: fixture.engine, modelId: 'test-model', maxRetries: 1 })
        : fixture.model

    await expect(model.invoke('Hi')).rejects.toMatchObject({
      name: 'AbortError',
      message: 'Generation cancelled.',
    })
    expect(fixture.create).toHaveBeenCalledOnce()
  },
)

it.each(['local', 'browser'] as const)(
  '%s honors a call timeout during a native callback',
  async (mode) => {
    const hold = pause()
    const onChunk = vi.fn().mockImplementationOnce(() => hold.wait)
    const fixture = browser({ onChunk })
    const model =
      mode === 'local'
        ? new ChatWebLLM({ engine: fixture.engine, modelId: 'test-model', onChunk })
        : fixture.model
    const pending = model.invoke('Hi', { timeout: 500 })
    const rejected = expect(pending).rejects.toHaveProperty('name', 'TimeoutError')
    try {
      await vi.waitFor(() => expect(onChunk).toHaveBeenCalledOnce(), { interval: 10 })
      await rejected
      expect((await model.invoke('Recover')).text).toBe('Answer')
    } finally {
      hold.resume()
      await pending.catch(() => {})
    }
  },
)

describe('local failed-attempt cancellation', () => {
  it('does not start a queued failed-attempt handler after cancellation', async () => {
    class TestCaller extends SignalCaller {
      fail(error: Error) {
        return this.onFailedAttempt!(error) as Promise<void>
      }
    }
    const controller = new AbortController()
    const onFailedAttempt = vi.fn()
    const caller = new TestCaller({ onFailedAttempt }, controller.signal)
    const pending = caller.fail(new Error('Opening failed'))
    controller.abort()

    await expect(pending).rejects.toBe(controller.signal.reason)
    expect(onFailedAttempt).not.toHaveBeenCalled()
  })

  it('keeps the default retry policy when a signal is present', async () => {
    const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const failure = Object.assign(new Error('Invalid request'), { status: 400 })
    create.mockRejectedValueOnce(failure)
    const model = new ChatWebLLM({ engine, modelId: 'test-model', maxRetries: 2 })

    await expect(model.invoke('Hi', { signal: new AbortController().signal })).rejects.toBe(failure)
    expect(getRetryable(failure)).toBe(false)
    expect(create).toHaveBeenCalledOnce()
    expect((await model.invoke('Recover')).text).toBe('Recovered')
  })

  it.each([0, 1])('releases a pending handler with maxRetries=%i', async (maxRetries) => {
    const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
    const hold = pause()
    const controller = new AbortController()
    const onFailedAttempt = vi.fn(async () => hold.wait)
    const model = new ChatWebLLM({
      engine,
      modelId: 'test-model',
      maxRetries,
      maxConcurrency: 1,
      onFailedAttempt,
    })
    create.mockRejectedValueOnce(new Error('Opening failed'))
    const pending = model.invoke('Hi', { signal: controller.signal })
    const rejected = expect(pending).rejects.toHaveProperty('name', 'AbortError')
    try {
      await vi.waitFor(() => expect(onFailedAttempt).toHaveBeenCalledOnce())
      controller.abort()
      await rejected
      expect((await model.invoke('Recover')).text).toBe('Recovered')
    } finally {
      hold.resume()
    }
    expect(create).toHaveBeenCalledTimes(2)
    expect(onFailedAttempt).toHaveBeenCalledOnce()
  })
})

describe('browser failed-attempt callbacks', () => {
  it('batches the successful response after a failed-attempt callback', async () => {
    const onFailedAttempt = vi.fn()
    const { page, model, create } = browser({ maxRetries: 1, onFailedAttempt })
    const evaluate = vi.spyOn(page, 'evaluate')
    create.mockRejectedValueOnce(new Error('Opening failed'))

    expect((await finishRetries(() => model.invoke('Hi'))).text).toBe('Answer')

    expect(onFailedAttempt).toHaveBeenCalledOnce()
    expect(create).toHaveBeenCalledTimes(2)
    expect(evaluate).toHaveBeenCalledTimes(4)
  })

  it('awaits the handler before retrying without changing native requests or chunks', async () => {
    const hold = pause()
    const native = stampRetryable(
      Object.assign(new Error('Opening failed', { cause: new TypeError('GPU busy') }), {
        status: 400,
        code: 'GPU_BUSY',
      }),
      true,
    )
    const onChunk = vi.fn()
    const onFailedAttempt = vi.fn(async (error: Error) => {
      expect(error).toMatchObject({
        message: native.message,
        status: 400,
        code: 'GPU_BUSY',
        cause: { name: 'TypeError', message: 'GPU busy' },
      })
      expect(getRetryable(error)).toBe(true)
      await hold.wait
    })
    const { model, create } = browser({ maxRetries: 1, onFailedAttempt, onChunk })
    create.mockRejectedValueOnce(native)
    const pending = model.invoke('Keep this input: 한글 <>&')
    try {
      await vi.waitFor(() => expect(onFailedAttempt).toHaveBeenCalledOnce())
      expect(create).toHaveBeenCalledOnce()
      expect(onChunk).not.toHaveBeenCalled()
    } finally {
      hold.resume()
    }
    expect((await pending).text).toBe('Answer')
    expect(create).toHaveBeenCalledTimes(2)
    expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
    expect(onChunk.mock.calls.map(([value]) => value)).toEqual([
      chunk({ content: 'Answer' }, 'stop'),
    ])
  })

  it('retries when a provider error refuses property inspection', async () => {
    const failure = new Proxy(new Error('GPU busy'), {
      getOwnPropertyDescriptor(target, key) {
        if (key === 'message') throw new Error('Cannot inspect provider fields')
        return Reflect.getOwnPropertyDescriptor(target, key)
      },
    })
    const onFailedAttempt = vi.fn()
    const { model, create } = browser({ maxRetries: 1, onFailedAttempt })
    create.mockRejectedValueOnce(failure)

    expect((await finishRetries(() => model.invoke('Hi'))).text).toBe('Answer')
    expect(onFailedAttempt).toHaveBeenCalledOnce()
    expect(onFailedAttempt.mock.calls[0]?.[0]).toMatchObject({ message: 'GPU busy' })
    expect(create).toHaveBeenCalledTimes(2)
  })

  it('handles the final failure and returns changed error fields and retry markers', async () => {
    const native = Object.assign(new Error('Opening failed'), { code: 'OLD', retryAfterMs: 1 })
    const onFailedAttempt = vi.fn((error: Error & { code?: string; retryAfterMs?: number }) => {
      error.message = 'Inspected failure'
      error.retryAfterMs = 20
      delete error.code
      stampRetryable(error, false)
    })
    const { model, create } = browser({ maxRetries: 0, onFailedAttempt })
    create.mockRejectedValueOnce(native)
    const error = await model.invoke('Hi').catch((error: unknown) => error)
    expect(onFailedAttempt).toHaveBeenCalledOnce()
    expect(error).toMatchObject({ message: 'Inspected failure', retryAfterMs: 20 })
    expect(error).not.toHaveProperty('code')
    expect(getRetryable(error)).toBe(false)
    expect(native.message).toBe('Inspected failure')
    expect(native.retryAfterMs).toBe(20)
    expect(native).not.toHaveProperty('code')
    expect(create).toHaveBeenCalledOnce()
    expect((await model.invoke('Recover')).text).toBe('Answer')
  })

  it.each([false, true])('stops and preserves the handler error (async=%s)', async (async) => {
    const stopped = new Error('Do not retry')
    const onFailedAttempt = vi.fn(() => {
      if (async) return Promise.reject(stopped)
      throw stopped
    })
    const { model, create } = browser({ maxRetries: 2, onFailedAttempt })
    create.mockRejectedValueOnce(new Error('Opening failed'))
    await expect(model.invoke('Hi')).rejects.toBe(stopped)
    expect(create).toHaveBeenCalledOnce()
    expect(onFailedAttempt).toHaveBeenCalledOnce()
    expect((await model.invoke('Recover')).text).toBe('Answer')
  })

  it.each([
    [0, 1],
    [2, 0],
  ])('uses %i default retries and %i call retries with the handler', async (defaults, retries) => {
    const onFailedAttempt = vi.fn()
    const { model, create } = browser({ maxRetries: defaults, onFailedAttempt })
    create.mockRejectedValueOnce(new Error('Opening failed'))
    if (retries)
      expect((await finishRetries(() => model.invoke('Hi', { maxRetries: retries }))).text).toBe(
        'Answer',
      )
    else await expect(model.invoke('Hi', { maxRetries: retries })).rejects.toThrow('Opening failed')
    expect(create).toHaveBeenCalledTimes(retries + 1)
    expect(onFailedAttempt).toHaveBeenCalledOnce()
  })

  it('releases a browser handler waiting for acknowledgment when the bridge closes', async () => {
    const { bridge, create } = browser()
    create.mockRejectedValueOnce(new Error('Opening failed'))
    const id = await bridge.start({
      config: { modelId: 'test-model', maxRetries: 2 },
      messages: [{ role: 'human', content: 'Hi' }],
      options: {},
      includeFailedAttempts: true,
    })
    expect(await bridge.next(id)).toMatchObject({
      failedAttempt: { message: 'Opening failed' },
    })
    await bridge.close()
    expect(create).toHaveBeenCalledOnce()
  })

  it('handles consecutive failed attempts in order', async () => {
    const seen: string[] = []
    const { model, create } = browser({
      maxRetries: 2,
      onFailedAttempt: (error: Error) => seen.push(error.message),
    })
    create
      .mockRejectedValueOnce(new Error('First failure'))
      .mockRejectedValueOnce(new Error('Second failure'))
    expect((await finishRetries(() => model.invoke('Hi'))).text).toBe('Answer')
    expect(seen).toEqual(['First failure', 'Second failure'])
    expect(create).toHaveBeenCalledTimes(3)
    expect(create.mock.calls.map(([request]) => request)).toEqual(
      Array(3).fill(create.mock.calls[0]?.[0]),
    )
  }, 10_000)

  it('cancels while the browser retry response is pending', async () => {
    const { page, bridge, create } = browser()
    const acknowledging = pause()
    const release = pause()
    const stop = vi.spyOn(bridge, 'stop')
    let count = 0
    const delayed: BrowserPage = {
      async evaluate(run, arg) {
        const value = await page.evaluate(run, arg)
        if (++count === 3) {
          acknowledging.resume()
          await release.wait
        }
        return value
      },
    }
    const model = new BrowserChatWebLLM(delayed, {
      modelId: 'test-model',
      maxRetries: 1,
      onFailedAttempt: () => {},
    })
    create.mockRejectedValueOnce(new Error('GPU busy'))
    const controller = new AbortController()
    const pending = model._streamResponseChunks([], { signal: controller.signal }).next()
    const failed = vi.fn()
    void pending.catch(failed)
    try {
      await acknowledging.wait
      controller.abort()
      await vi.waitFor(() => expect(failed).toHaveBeenCalledWith(controller.signal.reason))
      expect(stop).toHaveBeenCalledOnce()
    } finally {
      release.resume()
      await pending.catch(() => {})
    }
  })

  it('cancels a pending handler and ignores its late completion', async () => {
    const hold = pause()
    const controller = new AbortController()
    const onFailedAttempt = vi.fn(async () => hold.wait)
    const { model, create } = browser({ maxRetries: 2, onFailedAttempt })
    create.mockRejectedValueOnce(new Error('Opening failed'))
    const pending = model.invoke('Hi', { signal: controller.signal })
    const rejected = expect(pending).rejects.toHaveProperty('name', 'AbortError')
    try {
      await vi.waitFor(() => expect(onFailedAttempt).toHaveBeenCalledOnce())
      controller.abort()
      await rejected
      expect(create).toHaveBeenCalledOnce()
      expect((await model.invoke('Recover')).text).toBe('Answer')
    } finally {
      hold.resume()
    }
    await Promise.resolve()
    expect(create).toHaveBeenCalledTimes(2)
  })

  it('does not call the retry handler or replay content after a stream failure', async () => {
    const onFailedAttempt = vi.fn()
    const { model, create } = browser({ maxRetries: 1, onFailedAttempt })
    create.mockResolvedValueOnce(
      (async function* () {
        yield* chunks(chunk({ content: 'Original bytes' }))
        throw new Error('Decode failed')
      })(),
    )
    const text: string[] = []
    await expect(
      (async () => {
        for await (const chunk of await model.stream('Hi')) text.push(chunk.text)
      })(),
    ).rejects.toThrow('Decode failed')
    expect(text.join('')).toBe('Original bytes')
    expect(create).toHaveBeenCalledOnce()
    expect(onFailedAttempt).not.toHaveBeenCalled()
    expect((await model.invoke('Recover')).text).toBe('Answer')
  })
})
