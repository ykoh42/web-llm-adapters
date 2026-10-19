import { afterEach, describe, expect, it, vi } from 'vitest'
import { createAgent } from 'langchain'
import { AIMessage, AIMessageChunk, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import {
  BrowserChatWebLLM,
  BrowserWebLLMEmbeddings,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

function browser(...items: ReturnType<typeof chunk>[]) {
  const result = setup(...items)
  const bridge = createBrowserBridge(result.engine)
  vi.stubGlobal('window', { webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  return {
    ...result,
    bridge,
    page,
    model: new BrowserChatWebLLM(page, { modelId: 'Qwen3-8B-q4f16_1-MLC' }),
  }
}
afterEach(() => vi.unstubAllGlobals())

describe('browser transport', () => {
  it('hoists tool images through the browser transport into native vision input', async () => {
    const { page, create } = browser(chunk({ content: 'A picture.' }, 'stop'))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-vision-model' })
    const result = await model.invoke([
      new HumanMessage('Describe the picture.'),
      new AIMessage({ content: '', tool_calls: [{ id: 'picture', name: 'read', args: {} }] }),
      new ToolMessage({
        tool_call_id: 'picture',
        content: [{ type: 'image', base64: 'AA==', mime_type: 'image/png' }],
      }),
    ])

    expect(result.text).toBe('A picture.')
    expect(create.mock.calls[0]?.[0].messages.slice(-2)).toEqual([
      { role: 'tool', tool_call_id: 'picture', content: '[Image attached]' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }],
      },
    ])
  })

  it.each([
    [0, 1],
    [1, 0],
  ])('forwards %i model retries and %i call retries', async (defaults, retries) => {
    const { page, create } = browser(chunk({ content: 'Okay' }, 'stop'))
    create.mockRejectedValueOnce(new Error('Opening failed'))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model', maxRetries: defaults })

    const result = model.invoke('Hi', { maxRetries: retries })
    if (retries) expect((await result).text).toBe('Okay')
    else await expect(result).rejects.toThrow('Opening failed')
    expect(create).toHaveBeenCalledTimes(retries + 1)
  })

  it('keeps local embedding retries off when maxRetries is undefined', async () => {
    const { page, engine } = browser()
    const create = vi.spyOn(engine.embeddings, 'create').mockRejectedValue(new Error('GPU lost'))
    const options = { modelId: 'embedding-model' }
    Object.assign(options, { maxRetries: undefined })
    const embeddings = new BrowserWebLLMEmbeddings(page, options)

    await expect(embeddings.embedQuery('Orchid')).rejects.toThrow('GPU lost')
    expect(create).toHaveBeenCalledOnce()
  })

  it('forwards an agent response schema across the browser boundary', async () => {
    const { model, create } = browser(chunk({ content: '{"answer":42}' }, 'stop'))
    const agent = createAgent({ model, responseFormat: z.object({ answer: z.number() }) })

    const result = await agent.invoke({ messages: [{ role: 'user', content: 'Six times seven?' }] })

    expect(result.structuredResponse).toEqual({ answer: 42 })
    const format = create.mock.calls[0]?.[0].response_format
    expect(format?.type).toBe('json_object')
    expect(JSON.parse(format?.schema ?? '')).toMatchObject({
      type: 'object',
      properties: { answer: { type: 'number' } },
      required: ['answer'],
    })
  })

  it('forwards reasoning and tool calls through callbacks across the browser boundary', async () => {
    const { model } = browser(
      chunk(
        {
          content:
            '<think>Need the lookup.</think><tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>',
        },
        'stop',
      ),
    )
    const seen: AIMessageChunk[] = []
    await model
      .bindTools([
        {
          type: 'function',
          function: {
            name: 'lookup',
            parameters: { type: 'object', properties: { code: { type: 'string' } } },
          },
        },
      ])
      .invoke('Look up A7.', {
        callbacks: [
          {
            handleLLMNewToken(...args) {
              const chunk = args[5]?.chunk
              if (chunk && 'message' in chunk && AIMessageChunk.isInstance(chunk.message))
                seen.push(chunk.message)
            },
          },
        ],
      })
    expect(seen.flatMap((message) => message.contentBlocks)).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'Need the lookup.' }),
    )
    expect(seen.flatMap((message) => message.tool_calls ?? [])).toMatchObject([
      { name: 'lookup', args: { code: 'A7' } },
    ])
    expect(seen.at(-1)?.response_metadata.finish_reason).toBe('tool_calls')
  })

  it('runs the callback before yielding when the consumer stops early', async () => {
    const { model } = browser(chunk({ content: 'A' }), chunk({ content: 'B' }, 'stop'))
    const callback = vi.fn()
    const stream = await model.stream('Hi', { callbacks: [{ handleLLMNewToken: callback }] })

    expect((await stream.next()).value?.text).toBe('A')
    expect(callback).toHaveBeenCalledOnce()
    await stream.return()
  })

  it('passes image and audio messages directly to the browser bridge', async () => {
    const { model, page } = browser(chunk({ content: 'Okay' }, 'stop'))
    const evaluate = vi.spyOn(page, 'evaluate')
    const message = new HumanMessage({
      content: [
        { type: 'text', text: 'Describe these.' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
        { type: 'input_audio', input_audio: { format: 'wav', data: 'AQ==' } },
      ],
    })

    await model.invoke([message])

    expect(typeof evaluate.mock.calls[0]?.[1]).toBe('object')
    expect(evaluate.mock.calls[0]?.[1]).toMatchObject({ messages: [message.toDict()] })
  })

  it('explains how to expose the browser bridge when chat setup is missing', async () => {
    const { model } = browser()
    vi.stubGlobal('window', {})
    await expect(model.invoke('Hello')).rejects.toThrow(
      'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserChatWebLLM.',
    )
  })

  it('keeps overlapping bridge reads separate through the final event', async () => {
    const { bridge } = browser(chunk({ content: 'First' }), chunk({ content: 'Second' }, 'stop'))
    const id = await bridge.start({
      config: { modelId: 'test-model' },
      messages: [new HumanMessage('Hello').toDict()],
      options: {},
    })
    try {
      await bridge.next(id)
      const [second, final, done] = await Promise.all([
        bridge.next(id),
        bridge.next(id),
        bridge.next(id),
      ])
      expect(second).toMatchObject({ done: false, value: { text: 'Second' } })
      expect(final).toMatchObject({
        done: false,
        value: { message: { response_metadata: { finish_reason: 'stop' } } },
      })
      expect(done).toMatchObject({ done: true })
    } finally {
      await bridge.stop(id)
    }
  })

  it('runs a Node agent and preserves reasoning, tool IDs and tool results across the browser boundary', async () => {
    const { model, create } = browser()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Use lookup.</think><tool_call>\n{"name":"lookup","arguments":{"code":"A7"}}\n</tool_call>',
          },
          'stop',
        ),
      ),
    )
    create.mockResolvedValueOnce(chunks(chunk({ content: '<think>Done.</think>42' }, 'stop')))
    const lookup = tool(({ code }) => `Answer ${code}: 42`, {
      name: 'lookup',
      description: 'Get an answer.',
      schema: z.object({ code: z.string() }),
    })
    const agent = createAgent({ model, tools: [lookup] })
    const result = await agent.invoke({ messages: [{ role: 'user', content: 'Look up A7.' }] })
    expect(result.messages.at(-1)?.text).toBe('42')
    expect(result.messages[1]?.content).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'Use lookup.' }),
    )
    expect(JSON.stringify(create.mock.calls[1]?.[0])).toContain('Answer A7: 42')
  })

  it('returns native chunks and usage metadata without sending callbacks over JSON', async () => {
    const usage = {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 2,
        decode_tokens_per_s: 3,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.2,
      },
    }
    const native = { ...chunk({ content: 'Okay' }, 'stop'), usage }
    const { page } = browser(native)
    const onChunk = vi.fn()
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model', onChunk })
    const result = await model.invoke('Hi')
    expect(result.usage_metadata).toMatchObject({
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
    })
    expect(onChunk).toHaveBeenCalledExactlyOnceWith(native)
  })

  it('passes browser errors through and releases the failed stream', async () => {
    const { model, create, bridge } = browser(chunk({ content: 'Recovered' }, 'stop'))
    create.mockRejectedValueOnce(
      Object.assign(new Error('Context full'), { name: 'ContextWindowSizeExceededError' }),
    )
    await expect(model.invoke('Hi')).rejects.toMatchObject({
      name: 'ContextWindowSizeExceededError',
      message: 'Context full',
    })
    expect((await model.invoke('After')).text).toBe('Recovered')
    await bridge.close()
    await expect(model.invoke('Closed')).rejects.toThrow('closed')
    expect(await bridge.next('unknown')).toMatchObject({
      error: { message: 'Unknown browser call.' },
    })
  })

  it('keeps the stream error when browser cleanup also fails', async () => {
    const failure = new Error('Generation failed')
    let call = 0
    const page: BrowserPage = {
      async evaluate<T>(): Promise<T> {
        call++
        if (call === 1) return 'stream-id' as T
        if (call === 2) throw failure
        throw new Error('Page closed during cleanup')
      },
    }
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })

    await expect(model.invoke('Hello')).rejects.toBe(failure)
  })

  it('does not send a redundant stop after the browser stream ends', async () => {
    const { model, bridge } = browser(chunk({ content: 'Done' }, 'stop'))
    const stop = vi.spyOn(bridge, 'stop')

    await model.invoke('Hello')

    expect(stop).not.toHaveBeenCalled()
  })

  it('closes without unloading the caller engine', async () => {
    const { bridge, engine } = browser()
    const unload = vi.spyOn(engine, 'unload')
    await bridge.close()
    await bridge.close()
    expect(unload).not.toHaveBeenCalled()
    await expect(
      bridge.start({ config: { modelId: 'test' }, messages: [], options: {} }),
    ).rejects.toThrow('closed')
  })

  it('cancels browser generation and recovers on the next request', async () => {
    const { model, create, stop } = browser(chunk({ content: 'Recovered' }, 'stop'))
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'Started' })
        await hold.wait
        yield chunk({}, 'abort')
      })(),
    )
    stop.mockImplementation(async () => hold.resume())
    const cancel = new AbortController()
    const stream = await model.stream('Hi', { signal: cancel.signal })
    expect((await stream.next()).value?.text).toBe('Started')
    cancel.abort()
    await expect(collect(stream)).rejects.toBeDefined()
    expect((await model.invoke('After')).text).toBe('Recovered')
  })
})

it('uses browser embeddings in LangGraph memory and waits for native cleanup', async () => {
  const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
  const { InMemoryStore } = await import('@langchain/langgraph')
  const { page, engine, bridge } = browser()
  const evaluate = vi.spyOn(page, 'evaluate')
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => ({
    object: 'list',
    model: 'embedding',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: (request.input as string[]).map((text, index) => ({
      object: 'embedding',
      index,
      embedding: text.includes('orchid') ? [1, 0] : [0, 1],
    })),
  }))
  const embeddings = new BrowserWebLLMEmbeddings(page, {
    modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
  })
  expect(await embeddings.embedQuery('orchid')).toEqual([1, 0])
  expect(evaluate.mock.calls[0]?.[1]).toMatchObject({
    config: { modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4' },
    texts: ['orchid'],
    type: 'query',
  })
  expect(create.mock.calls[0]?.[0].input).toEqual([
    'Represent this sentence for searching relevant passages: orchid',
  ])
  const store = new InMemoryStore({ index: { embeddings, dims: 2 } })
  await store.put(['notes'], 'flower', { text: 'An orchid.' })
  await store.put(['notes'], 'car', { text: 'A car.' })
  expect((await store.search(['notes'], { query: 'orchid', limit: 1 }))[0]?.key).toBe('flower')
  const hold = pause()
  create.mockImplementationOnce(async () => {
    await hold.wait
    return {
      object: 'list',
      model: 'embedding',
      usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      data: [{ object: 'embedding', index: 0, embedding: [1, 0] }],
    }
  })
  const pending = embeddings.embedQuery('orchid')
  await new Promise((resolve) => setTimeout(resolve, 0))
  let closed = false
  const closing = bridge.close().then(() => {
    closed = true
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(closed).toBe(false)
  hold.resume()
  await pending
  await closing
  await expect(embeddings.embedQuery('orchid')).rejects.toThrow('closed')
})

it('bounds browser embedding requests with the configured batch size', async () => {
  const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
  const { page, engine, bridge } = browser()
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => {
    const documents = request.input as string[]
    return {
      object: 'list',
      model: 'embedding-model',
      usage: {
        prompt_tokens: documents.length,
        total_tokens: documents.length,
        extra: { prefill_tokens_per_s: 1 },
      },
      data: documents.map((text, index) => ({
        object: 'embedding',
        index,
        embedding: [Number(text.slice(8)), 1],
      })),
    }
  })
  const embeddings = new BrowserWebLLMEmbeddings(page, {
    modelId: 'embedding-model',
    batchSize: 2,
  })
  const documents = Array.from({ length: 5 }, (_, index) => `Document ${index}`)

  const result = await embeddings.embedDocuments(documents)

  expect(create.mock.calls.map(([request]) => request.input)).toEqual([
    ['Document 0', 'Document 1'],
    ['Document 2', 'Document 3'],
    ['Document 4'],
  ])
  expect(result).toEqual(documents.map((_, index) => [index, 1]))
  await bridge.close()
})

it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
  'rejects invalid browser embedding batch size %s',
  async (batchSize) => {
    const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
    const { page } = browser()

    expect(
      () => new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', batchSize }),
    ).toThrow('batchSize must be a positive integer.')
  },
)

it('explains how to expose the browser bridge when embedding setup is missing', async () => {
  const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
  const { page } = browser()
  vi.stubGlobal('window', {})
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })
  await expect(embeddings.embedQuery('hello')).rejects.toThrow(
    'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserWebLLMEmbeddings.',
  )
})

it('handles empty embedding inputs before crossing the browser boundary', async () => {
  const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
  const { page, bridge } = browser()
  const evaluate = vi.spyOn(page, 'evaluate')
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })

  await expect(embeddings.embedDocuments([])).resolves.toEqual([])
  await expect(embeddings.embedDocuments(['  '])).rejects.toThrow('WebLLM cannot embed empty text.')
  await expect(embeddings.embedQuery(' \t')).rejects.toThrow('WebLLM cannot embed empty text.')
  expect(evaluate).not.toHaveBeenCalled()
  await bridge.close()
})

it('snapshots browser embedding inputs before a delayed page transfer', async () => {
  const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
  const { engine, bridge } = browser()
  const hold = pause()
  const page: BrowserPage = {
    async evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T> {
      await hold.wait
      const result = await run(structuredClone(arg))
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => ({
    object: 'list',
    model: 'embedding-model',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: (request.input as string[]).map((text, index) => ({
      object: 'embedding',
      index,
      embedding: text === 'Before' ? [1, 0] : [0, 1],
    })),
  }))
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })
  const documents = ['Before']
  const pending = embeddings.embedDocuments(documents)
  documents[0] = 'After'
  hold.resume()

  await expect(pending).resolves.toEqual([[1, 0]])
  expect(create.mock.calls[0]?.[0].input).toEqual(['Before'])
  await bridge.close()
})
