import { expect, it } from 'vitest'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { HumanMessage } from '@langchain/core/messages'
import { Message, TextBlock } from '@strands-agents/sdk'
// Package exports load separate release bundles, each with its own copy of the queue code.
import { WebLLMModel, withEngine as strandsAccess } from 'strands-web-llm'
import { ChatWebLLM, withEngine as langchainAccess } from 'langchain-web-llm'
import { chunk, pause, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

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
