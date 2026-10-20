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
import { chunk, chunks, pause, setup } from '../fixtures/engine.js'

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
  return { ...result, model, bridge }
}
afterEach(async () => {
  try {
    await Promise.all(bridges.splice(0).map((bridge) => bridge.close()))
  } finally {
    vi.unstubAllGlobals()
  }
})

describe('local failed-attempt cancellation', () => {
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
    const pending = model.invoke('Hi', { maxRetries: retries })
    if (retries) expect((await pending).text).toBe('Answer')
    else await expect(pending).rejects.toThrow('Opening failed')
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
    expect((await model.invoke('Hi')).text).toBe('Answer')
    expect(seen).toEqual(['First failure', 'Second failure'])
    expect(create).toHaveBeenCalledTimes(3)
    expect(create.mock.calls.map(([request]) => request)).toEqual(
      Array(3).fill(create.mock.calls[0]?.[0]),
    )
  }, 10_000)

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
