import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { BrowserModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  BrowserWebLLMEmbeddings,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
  type ChatWebLLMOptions,
} from 'langchain-web-llm'
import { chunk, collect, pause, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

// Playwright drops reserved keys in both directions; ordinary JSON.parse does not.
function transfer<T>(input: T): T {
  const copy = structuredClone(input)
  const pending: unknown[] = [copy]
  while (pending.length) {
    const value = pending.pop()
    if (!value || typeof value !== 'object') continue
    for (const key of Object.keys(value)) {
      if (key === '__proto__') Reflect.deleteProperty(value, key)
      else pending.push((value as Record<string, unknown>)[key])
    }
  }
  return copy
}

function open(
  client: 'strands' | 'langchain' | 'embeddings',
  failure: Error,
  options: Pick<ChatWebLLMOptions, 'onFailedAttempt' | 'onChunk'> = {},
) {
  const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
  if (client === 'embeddings')
    vi.spyOn(engine.embeddings, 'create')
      .mockRejectedValueOnce(failure)
      .mockResolvedValue({
        object: 'list',
        model: 'test-model',
        data: [{ object: 'embedding', index: 0, embedding: [1, 0] }],
        usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      })
  else create.mockRejectedValueOnce(failure)
  const bridge = client === 'strands' ? strandsBridge(engine) : langchainBridge(engine)
  vi.stubGlobal('window', { webllmBridge: bridge, webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      return transfer(await run(transfer(arg)))
    },
  }
  let invoke: () => Promise<unknown>
  if (client === 'strands') {
    const model = new BrowserModel(page, {
      modelId: 'test-model',
      ...(options.onChunk && { onChunk: options.onChunk }),
    })
    invoke = () =>
      collect(model.stream([new Message({ role: 'user', content: [new TextBlock('Hello')] })]))
  } else if (client === 'embeddings') {
    const model = new BrowserWebLLMEmbeddings(page, { modelId: 'test-model' })
    invoke = () => model.embedQuery('Hello')
  } else {
    const model = new BrowserChatWebLLM(page, {
      modelId: 'test-model',
      ...options,
    })
    invoke = () => model.invoke('Hello')
  }
  return { bridge, invoke, create }
}

const causes = [
  '{"__proto__":{"kept":true},"constructor":"data"}',
  '{"items":[{"__proto__":{"kept":true},"value":42}]}',
]

it.each(['langchain', 'strands', 'embeddings'] as const)(
  '%s keeps local error edits in its browser-safe snapshot',
  async (client) => {
    const { bridge } = open(client, new Error('Unused native failure'))
    try {
      const packet =
        client === 'embeddings' && 'embed' in bridge
          ? await bridge.embed({ config: { modelId: 'test-model' }, texts: [], type: 'query' })
          : await bridge.next('unknown')
      if (!('error' in packet)) throw new Error('Expected a browser error.')
      packet.error.message = 'Edited locally'
      packet.error.causeValue = { value: JSON.parse(causes[0]!) }
      expect(JSON.parse(packet.errorJSON!)).toEqual(packet.error)
    } finally {
      await bridge.close()
    }
  },
)

it('accepts direct failed-attempt edits without using a stale snapshot', async () => {
  const native = new TypeError('Native failure', { cause: 'Original cause' })
  const { bridge } = open('langchain', native)
  if (!('retry' in bridge)) throw new Error('Expected the LangChain bridge.')
  try {
    const id = await bridge.start({
      config: { modelId: 'test-model', maxRetries: 0 },
      messages: [{ role: 'human', content: 'Hello' }],
      options: {},
      includeFailedAttempts: true,
    })
    const packet = await bridge.next(id)
    if (!('failedAttempt' in packet)) throw new Error('Expected a failed attempt.')
    const changed: unknown = JSON.parse(causes[1]!)
    packet.failedAttempt.message = 'Edited locally'
    packet.failedAttempt.causeValue = { value: JSON.parse(causes[1]!) }
    expect(JSON.parse(packet.failedAttemptJSON!)).toEqual(packet.failedAttempt)
    bridge.retry(id, packet.failedAttempt)
    expect(await bridge.next(id)).toMatchObject({ error: { message: 'Edited locally' } })
    expect(native.message).toBe('Edited locally')
    expect(native.cause).toEqual(changed)
  } finally {
    await bridge.close()
  }
})

for (const client of ['strands', 'langchain', 'embeddings'] as const) {
  it.each(causes)(
    `${client} preserves reserved cause keys through browser transport: %s`,
    async (text) => {
      const cause: unknown = JSON.parse(text)
      const native = new TypeError('Native failure', { cause })
      const { bridge, invoke } = open(client, native)
      try {
        const failure = await invoke().catch((error: unknown) => error)
        expect(failure).toBeInstanceOf(TypeError)
        expect(failure).toHaveProperty('cause', cause)
        expect(Object.getPrototypeOf(failure)).toBe(TypeError.prototype)
        expect(JSON.stringify(await invoke())).toContain(
          client === 'embeddings' ? '1,0' : 'Recovered',
        )
      } finally {
        await bridge.close()
      }
    },
  )
}

for (const client of ['langchain', 'embeddings'] as const) {
  it(`${client} preserves reserved provider fields without changing error prototypes`, async () => {
    const metadata: unknown = JSON.parse(causes[1]!)
    const native = Object.assign(new TypeError('Native failure'), { metadata })
    Object.defineProperty(native, '__proto__', {
      value: { provider: true },
      enumerable: true,
      configurable: true,
      writable: true,
    })
    const { bridge, invoke } = open(client, native)
    try {
      const failure = await invoke().catch((error: unknown) => error)
      expect(failure).toHaveProperty('metadata', metadata)
      expect(Object.getOwnPropertyDescriptor(failure, '__proto__')?.value).toEqual({
        provider: true,
      })
      expect(Object.getPrototypeOf(failure)).toBe(TypeError.prototype)
      expect(Object.prototype).not.toHaveProperty('provider')
    } finally {
      await bridge.close()
    }
  })
}

it.each([false, true])(
  'preserves failed-attempt keys on both sides (edited=%s)',
  async (edited) => {
    const initial: unknown = JSON.parse(causes[0]!)
    const changed: unknown = JSON.parse(causes[1]!)
    const native = Object.assign(new TypeError('Native failure', { cause: initial }), {
      metadata: initial,
    })
    const handler = vi.fn((error: Error & { metadata: unknown }) => {
      expect(error.cause).toEqual(initial)
      expect(error.metadata).toEqual(initial)
      if (edited) error.cause = error.metadata = changed
    })
    const { bridge, invoke } = open('langchain', native, { onFailedAttempt: handler })
    try {
      const failure = await invoke().catch((error: unknown) => error)
      expect(handler).toHaveBeenCalledOnce()
      for (const error of [failure, native]) {
        expect(error).toHaveProperty('cause', edited ? changed : initial)
        expect(error).toHaveProperty('metadata', edited ? changed : initial)
        expect(Object.getPrototypeOf(error)).toBe(TypeError.prototype)
      }
      expect(JSON.stringify(await invoke())).toContain('Recovered')
    } finally {
      await bridge.close()
    }
  },
)

it.each(['langchain', 'strands'] as const)(
  '%s keeps raw chunk values beside an error snapshot',
  async (client) => {
    const hold = pause()
    vi.stubGlobal('scheduler', { yield: () => hold.wait })
    const native = chunk({})
    native.choices[0]!.logprobs = {
      content: [{ token: 'x', logprob: -Infinity, bytes: null, top_logprobs: [] }],
    }
    const failure = new TypeError('Native failure', { cause: JSON.parse(causes[0]!) })
    const onChunk = vi.fn()
    const { bridge, invoke, create } = open(client, failure, { onChunk })
    create.mockReset().mockResolvedValueOnce(
      (async function* () {
        yield native
        throw failure
      })(),
    )
    try {
      await expect(invoke()).rejects.toThrow('Native failure')
      expect(onChunk).toHaveBeenCalledExactlyOnceWith(native)
    } finally {
      hold.resume()
      await bridge.close()
    }
  },
)
