import { expect, it, vi } from 'vitest'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { HumanMessage } from '@langchain/core/messages'
import { Message, TextBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
// Package exports load separate release bundles, each with its own copy of the queue code.
import { WebLLMModel, withEngine as strandsAccess } from 'strands-web-llm'
import { ChatWebLLM, withEngine as langchainAccess } from 'langchain-web-llm'
import { createWebLLMProvider, withEngine as piAccess } from 'pi-ai-web-llm'
import { chunk, chunks, pause, setupEngine } from '../../fixtures/web-llm.js'

const clients = ['strands', 'langchain', 'pi'] as const
const access = { strands: strandsAccess, langchain: langchainAccess, pi: piAccess }
type Client = (typeof clients)[number]
const pairs = clients.flatMap((client) =>
  clients.filter((other) => client !== other).map((other) => ({ client, other })),
)

async function read(
  client: Client,
  engine: MLCEngineInterface,
  text: string,
  signal?: AbortSignal,
  onChunk?: () => void,
  onStop?: (reason: string) => void,
) {
  let output = ''
  if (client === 'strands') {
    const model = new WebLLMModel({ engine, modelId: 'test-model', ...(onChunk && { onChunk }) })
    const messages = [new Message({ role: 'user', content: [new TextBlock(text)] })]
    for await (const event of model.stream(messages, signal ? { cancelSignal: signal } : {})) {
      if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
        output += event.delta.text
      if (event.type === 'modelMessageStopEvent') onStop?.(event.stopReason)
    }
  } else if (client === 'langchain') {
    const model = new ChatWebLLM({ engine, modelId: 'test-model', ...(onChunk && { onChunk }) })
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage(text)],
      signal ? { signal } : {},
    ))
      output += chunk.text
  } else {
    const provider = createWebLLMProvider({
      engine,
      modelId: 'test-model',
      contextWindow: 8192,
      ...(onChunk && { onChunk }),
    })
    const model = provider.getModels()[0]!
    const stream = provider.stream(
      model,
      normalizeContext({ messages: [{ role: 'user', content: text, timestamp: 0 }] }),
      signal ? { signal } : {},
    )
    for await (const event of stream) if (event.type === 'text_delta') output += event.delta
    const result = await stream.result()
    if (result.stopReason === 'aborted') {
      signal?.throwIfAborted()
      throw new Error(result.errorMessage)
    }
    if (result.stopReason === 'error') throw new Error(result.errorMessage)
  }
  return output
}

function settle(pending: Promise<string>) {
  return pending.then(
    (text) => ({ text }),
    (error: unknown) => ({ error }),
  )
}

function turn() {
  return new Promise<void>((resolve) => setImmediate(resolve))
}

it('cancels a shared signal across independent engines and leaves each usable', async () => {
  const controller = new AbortController()
  const strandsStop = vi.fn()
  const entries = clients.map((client) => {
    const native = setupEngine(chunk({ content: 'recovered' }, 'stop'))
    const opened = pause()
    const hold = pause()
    native.create.mockResolvedValueOnce(
      (async function* () {
        opened.resume()
        yield chunk({ content: 'active' })
        await hold.wait
        yield chunk({}, 'stop')
      })(),
    )
    native.stop.mockImplementation(async () => hold.resume())
    return { client, ...native, opened, hold }
  })
  const active = entries.map(({ client, engine }) =>
    settle(read(client, engine, 'active', controller.signal, undefined, strandsStop)),
  )
  try {
    await Promise.all(entries.map(({ opened }) => opened.wait))
    const reason = new Error('Cancel all calls')
    controller.abort(reason)
    const results = await Promise.all(active)
    expect(results[0]).toMatchObject({ text: expect.any(String) })
    expect(strandsStop).toHaveBeenCalledExactlyOnceWith('cancelled')
    expect(results.slice(1)).toEqual([{ error: reason }, { error: reason }])
    for (const { client, engine, stop } of entries) {
      expect(stop).toHaveBeenCalledOnce()
      expect(await read(client, engine, 'later')).toBe('recovered')
    }
  } finally {
    for (const { hold } of entries) hold.resume()
    await Promise.all(active)
  }
})

for (const { client, other } of pairs) {
  it(`${client} leaves an independent ${other} engine free to run`, async () => {
    const first = setupEngine()
    const second = setupEngine(chunk({ content: 'independent' }, 'stop'))
    const opened = pause()
    const hold = pause()
    first.create.mockResolvedValueOnce(
      (async function* () {
        opened.resume()
        yield chunk({ content: 'active' })
        await hold.wait
        yield chunk({}, 'stop')
      })(),
    )
    const active = settle(read(client, first.engine, 'active'))
    await opened.wait
    const independent = settle(read(other, second.engine, 'independent'))
    try {
      await turn()
      expect(second.create).toHaveBeenCalledTimes(1)
      expect(await independent).toEqual({ text: 'independent' })
      expect(first.stop).not.toHaveBeenCalled()
    } finally {
      hold.resume()
      await Promise.all([active, independent])
    }
    expect(await active).toEqual({ text: 'active' })
  })

  it(`${client} owns the engine until its stream closes before ${other} starts`, async () => {
    const { engine, create } = setupEngine(chunk({ content: 'queued' }, 'stop'))
    const opened = pause()
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        opened.resume()
        yield chunk({ content: 'active' })
        await hold.wait
        yield chunk({}, 'stop')
      })(),
    )
    const active = settle(read(client, engine, 'active'))
    await opened.wait
    const waiting = settle(read(other, engine, 'queued'))
    try {
      await turn()
      expect(create).toHaveBeenCalledTimes(1)
      hold.resume()
      expect(await active).toEqual({ text: 'active' })
      expect(await waiting).toEqual({ text: 'queued' })
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      hold.resume()
      await Promise.all([active, waiting])
    }
  })

  it(`${other} can cancel its waiter without interrupting an active ${client} stream`, async () => {
    const { engine, create, stop } = setupEngine(chunk({ content: 'later' }, 'stop'))
    const opened = pause()
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        opened.resume()
        yield chunk({ content: 'active' })
        await hold.wait
        yield chunk({}, 'stop')
      })(),
    )
    const active = settle(read(client, engine, 'active'))
    await opened.wait
    const controller = new AbortController()
    const waiting = settle(read(other, engine, 'queued', controller.signal))
    try {
      await turn()
      const failure = new Error('Cancel only the waiter')
      controller.abort(failure)
      expect(await waiting).toEqual({ error: failure })
      expect(create).toHaveBeenCalledTimes(1)
      expect(stop).not.toHaveBeenCalled()
      hold.resume()
      expect(await active).toEqual({ text: 'active' })
      expect(await read(other, engine, 'later')).toBe('later')
    } finally {
      hold.resume()
      await Promise.all([active, waiting])
    }
  })

  it(`${client} releases a failed stream before queued ${other} work starts`, async () => {
    const { engine, create, stop } = setupEngine(chunk({ content: 'recovered' }, 'stop'))
    const opened = pause()
    const hold = pause()
    const failure = new Error('Native stream failed')
    create.mockResolvedValueOnce(
      (async function* () {
        opened.resume()
        yield chunk({ content: 'partial' })
        await hold.wait
        throw failure
      })(),
    )
    const active = settle(read(client, engine, 'active'))
    await opened.wait
    const waiting = settle(read(other, engine, 'recovered'))
    try {
      await turn()
      expect(create).toHaveBeenCalledTimes(1)
      hold.resume()
      expect(await active).toEqual({ error: failure })
      expect(await waiting).toEqual({ text: 'recovered' })
      expect(stop).toHaveBeenCalledTimes(1)
    } finally {
      hold.resume()
      await Promise.all([active, waiting])
    }
  })

  it(`${client} closes a failed observer stream before queued ${other} work starts`, async () => {
    const { engine, create, stop } = setupEngine(chunk({ content: 'recovered' }, 'stop'))
    const closing = pause()
    const hold = pause()
    const failure = new Error('Observer failed')
    let closed = false
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]() {
        return {
          async next() {
            return { done: false, value: chunk({ content: 'partial' }) }
          },
          async return() {
            closing.resume()
            await hold.wait
            closed = true
            return { done: true, value: undefined }
          },
        }
      },
    })
    const active = settle(
      read(client, engine, 'active', undefined, () => {
        throw failure
      }),
    )
    await closing.wait
    const waiting = settle(read(other, engine, 'recovered'))
    try {
      await turn()
      expect(create).toHaveBeenCalledTimes(1)
      expect(stop).toHaveBeenCalledTimes(1)
      expect(closed).toBe(false)
      hold.resume()
      expect(await active).toMatchObject({ error: { message: failure.message } })
      expect(await waiting).toEqual({ text: 'recovered' })
      expect(closed).toBe(true)
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      hold.resume()
      await Promise.all([active, waiting])
    }
  })

  it(`${client} closes a cancelled opening stream before queued ${other} work starts`, async () => {
    const { engine, create, stop } = setupEngine(chunk({ content: 'recovered' }, 'stop'))
    const opening = pause()
    const started = pause()
    const closing = pause()
    const hold = pause()
    const close = vi.fn(async () => {
      closing.resume()
      await hold.wait
      return { done: true as const, value: undefined }
    })
    create.mockImplementationOnce(async () => {
      started.resume()
      await opening.wait
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: false as const, value: chunk({}, 'stop') }),
          return: close,
        }),
      }
    })
    const controller = new AbortController()
    const active = settle(read(client, engine, 'active', controller.signal))
    await started.wait
    const failure = new Error('Cancel while opening')
    controller.abort(failure)
    const waiting = settle(read(other, engine, 'recovered'))
    try {
      await turn()
      expect(create).toHaveBeenCalledTimes(1)
      opening.resume()
      await closing.wait
      await turn()
      expect(close).toHaveBeenCalledTimes(1)
      expect(create).toHaveBeenCalledTimes(1)
      expect(stop).toHaveBeenCalledTimes(1)
      hold.resume()
      expect(await active).toEqual({ error: failure })
      expect(await waiting).toEqual({ text: 'recovered' })
      expect(close).toHaveBeenCalledTimes(1)
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      opening.resume()
      hold.resume()
      await Promise.all([active, waiting])
    }
  })

  it(`${client} reserves the engine for a cancelled read before queued ${other} work starts`, async () => {
    const { engine, create } = setupEngine(chunk({ content: 'recovered' }, 'stop'))
    const reading = pause()
    const release = pause()
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          reading.resume()
          await release.wait
          return { done: false as const, value: chunk({ content: 'late' }, 'stop') }
        },
        return: async () => ({ done: true as const, value: undefined }),
      }),
    })
    const controller = new AbortController()
    const active = settle(read(client, engine, 'active', controller.signal))
    const finished = vi.fn()
    void active.then(finished)
    await reading.wait
    controller.abort(new Error('Stopped'))
    const waiting = settle(read(other, engine, 'recovered'))
    try {
      await vi.waitFor(() => expect(finished).toHaveBeenCalledOnce())
      expect(await active).not.toEqual({ text: 'late' })
      expect(create).toHaveBeenCalledOnce()
      release.resume()
      expect(await waiting).toEqual({ text: 'recovered' })
      expect(create).toHaveBeenCalledTimes(2)
    } finally {
      release.resume()
      await Promise.all([active, waiting])
    }
  })

  it(`${client} and ${other} keep queued requests in order after several cancellations`, async () => {
    const { engine, create, stop } = setupEngine()
    create.mockImplementation(async (request) =>
      chunks(chunk({ content: String(request.messages.at(-1)?.content) }, 'stop')),
    )
    const opened = pause()
    const hold = pause()
    const active = strandsAccess(engine, async () => {
      opened.resume()
      await hold.wait
    })
    await opened.wait
    const jobs: { controller: AbortController; result: ReturnType<typeof settle>; text: string }[] =
      []
    try {
      for (let index = 0; index < 12; index++) {
        const controller = new AbortController()
        const text = 'Request ' + index
        jobs.push({
          controller,
          text,
          result: settle(read(index % 2 ? client : other, engine, text, controller.signal)),
        })
        await turn()
      }
      const cancelled = [1, 2, 5, 8]
      const reasons = [new Error('Cancel waiter'), 'Cancelled', null, 0]
      for (const [index, position] of cancelled.entries()) {
        jobs[position]!.controller.abort(reasons[index])
        const result = await jobs[position]!.result
        expect('error' in result).toBe(true)
        if ('error' in result) expect(result.error).toBe(reasons[index])
      }
      expect(create).not.toHaveBeenCalled()
      expect(stop).not.toHaveBeenCalled()
      hold.resume()
      await active
      for (const [index, job] of jobs.entries()) {
        if (!cancelled.includes(index)) expect(await job.result).toEqual({ text: job.text })
      }
      const expected = jobs.filter((_, index) => !cancelled.includes(index)).map((job) => job.text)
      expect(create.mock.calls.map(([request]) => request.messages.at(-1)?.content)).toEqual(
        expected,
      )
      expect(stop).not.toHaveBeenCalled()
      expect(await read(client, engine, 'Recovered')).toBe('Recovered')
    } finally {
      hold.resume()
      await Promise.all([active, ...jobs.map((job) => job.result)])
    }
  })

  it(`${client} native access shares the release-bundle queue with ${other}`, async () => {
    const { engine, create } = setupEngine(chunk({ content: 'queued' }, 'stop'))
    const opened = pause()
    const hold = pause()
    const active = settle(
      access[client](engine, async () => {
        opened.resume()
        await hold.wait
        return 'native'
      }),
    )
    await opened.wait
    const waiting = settle(read(other, engine, 'queued'))
    try {
      await turn()
      expect(create).not.toHaveBeenCalled()
      hold.resume()
      expect(await active).toEqual({ text: 'native' })
      expect(await waiting).toEqual({ text: 'queued' })
    } finally {
      hold.resume()
      await Promise.all([active, waiting])
    }
  })
}
