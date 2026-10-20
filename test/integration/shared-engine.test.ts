import { expect, it } from 'vitest'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { HumanMessage } from '@langchain/core/messages'
import { Message, TextBlock } from '@strands-agents/sdk'
// Package exports load separate release bundles, each with its own copy of the queue code.
import { WebLLMModel, withEngine as strandsAccess } from 'strands-web-llm'
import { ChatWebLLM, withEngine as langchainAccess } from 'langchain-web-llm'
import { chunk, chunks, pause, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

type Client = 'strands' | 'langchain'

async function read(
  client: Client,
  engine: MLCEngineInterface,
  text: string,
  signal?: AbortSignal,
) {
  let output = ''
  if (client === 'strands') {
    const model = new WebLLMModel({ engine, modelId: 'test-model' })
    const messages = [new Message({ role: 'user', content: [new TextBlock(text)] })]
    for await (const event of model.stream(messages, signal ? { cancelSignal: signal } : {}))
      if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
        output += event.delta.text
  } else {
    const model = new ChatWebLLM({ engine, modelId: 'test-model' })
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage(text)],
      signal ? { signal } : {},
    ))
      output += chunk.text
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

for (const client of ['strands', 'langchain'] as const) {
  const other = client === 'strands' ? 'langchain' : 'strands'

  it(`${client} owns the engine until its stream closes before ${other} starts`, async () => {
    const { engine, create } = setup(chunk({ content: 'queued' }, 'stop'))
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
    const { engine, create, stop } = setup(chunk({ content: 'later' }, 'stop'))
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
    const { engine, create, stop } = setup(chunk({ content: 'recovered' }, 'stop'))
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

  it(`${client} and ${other} keep queued requests in order after several cancellations`, async () => {
    const { engine, create, stop } = setup()
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
    const { engine, create } = setup(chunk({ content: 'queued' }, 'stop'))
    const opened = pause()
    const hold = pause()
    const access = client === 'strands' ? strandsAccess : langchainAccess
    const active = settle(
      access(engine, async () => {
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
