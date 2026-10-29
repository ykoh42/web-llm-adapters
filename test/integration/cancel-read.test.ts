import { expect, it, vi } from 'vitest'
import { createModels } from '@earendil-works/pi-ai/models'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, chunks, collect, pause, setupEngine } from '../../fixtures/web-llm.js'

it.each(['strands', 'langchain'] as const)(
  '%s returns a closed stream after cancellation without releasing the busy engine',
  async (client) => {
    const { engine, create, stop } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
    const closing = pause()
    stop.mockImplementation(() => closing.wait)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'First' })))
    const controller = new AbortController()
    const stream =
      client === 'strands'
        ? new WebLLMModel({ engine, modelId: 'test-model' }).stream([], {
            cancelSignal: controller.signal,
          })
        : new ChatWebLLM({ engine, modelId: 'test-model' })._streamResponseChunks([], {
            signal: controller.signal,
          })
    while (!create.mock.calls.length) expect((await stream.next()).done).toBe(false)
    controller.abort()
    const returned = stream.return(undefined)
    const settled = vi.fn()
    void returned.then(settled, settled)
    const recovered = new ChatWebLLM({ engine, modelId: 'test-model' }).invoke('Next')
    try {
      await vi.waitFor(() => expect(settled).toHaveBeenCalledOnce())
      expect(create).toHaveBeenCalledOnce()
    } finally {
      closing.resume()
      await returned
    }
    expect((await recovered).text).toBe('Recovered')
    expect(create).toHaveBeenCalledTimes(2)
  },
)

it.each(['strands', 'langchain', 'pi'] as const)(
  '%s returns cancellation while a WebLLM read is pending',
  async (client) => {
    const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
    const reading = pause()
    const release = pause()
    const closed = vi.fn(async () => ({ done: true as const, value: undefined }))
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          reading.resume()
          await release.wait
          return { done: true as const, value: undefined }
        },
        return: closed,
      }),
    })
    const signal = new AbortController()
    let result: Promise<unknown>
    if (client === 'strands') {
      const model = new WebLLMModel({ engine, modelId: 'test-model' })
      result = collect(model.stream([], { cancelSignal: signal.signal }))
    } else if (client === 'langchain') {
      const model = new ChatWebLLM({ engine, modelId: 'test-model' })
      result = model.invoke('Hello', { signal: signal.signal }).catch((error: unknown) => error)
    } else {
      const provider = createWebLLMProvider({ engine, modelId: 'test-model', contextWindow: 8192 })
      const models = createModels()
      models.setProvider(provider)
      result = models.complete(
        provider.getModels()[0]!,
        { messages: [{ role: 'user', content: 'Hello', timestamp: 0 }] },
        { signal: signal.signal },
      )
    }
    const settled = vi.fn()
    void result.then(settled, settled)
    await reading.wait
    signal.abort()
    try {
      await vi.waitFor(() => expect(settled).toHaveBeenCalledOnce())
      const value: unknown = await result
      if (client === 'strands')
        expect(value).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
      else if (client === 'langchain') expect(value).toHaveProperty('name', 'AbortError')
      else expect(value).toHaveProperty('stopReason', 'aborted')
      expect(closed).toHaveBeenCalledOnce()
    } finally {
      release.resume()
    }
    const recovered = new ChatWebLLM({ engine, modelId: 'test-model' })
    expect((await recovered.invoke('Next')).text).toBe('Recovered')
  },
)
