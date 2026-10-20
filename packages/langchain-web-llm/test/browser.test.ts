import { afterEach, describe, expect, it, vi } from 'vitest'
import { ContextOverflowError, getRetryable } from '@langchain/core/errors'
import { createAgent } from 'langchain'
import { AIMessage, AIMessageChunk, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { CallbackManager } from '@langchain/core/callbacks/manager'
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
  it('delivers native tool chunks while the model is still generating arguments', async () => {
    const { page, bridge, create } = browser()
    const hold = pause()
    const onChunk = vi.fn()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: '<tool_call>' })
        yield chunk({ content: '{"name":"lookup","arguments":{' })
        await hold.wait
        yield chunk({ content: '}}</tool_call>' }, 'stop')
      })(),
    )
    const model = new BrowserChatWebLLM(page, { modelId: 'Qwen3-8B-q4f16_1-MLC', onChunk })
    const pending = model
      .bindTools([{ type: 'function', function: { name: 'lookup' } }])
      .invoke('Look up a note.')
    try {
      await vi.waitFor(() => expect(onChunk).toHaveBeenCalledTimes(2))
    } finally {
      hold.resume()
      await pending
      await bridge.close()
    }
    expect(onChunk).toHaveBeenCalledTimes(3)
    expect((await pending).tool_calls).toMatchObject([{ name: 'lookup', args: {} }])
  })

  it.each(['abort', 'throw'])(
    'closes a pending native tool stream when its callback %s',
    async (mode) => {
      const {
        page,
        bridge,
        create,
        stop,
        model: plain,
      } = browser(chunk({ content: 'Recovered' }, 'stop'))
      const hold = pause()
      const closed = vi.fn()
      const controller = new AbortController()
      create.mockResolvedValueOnce(
        (async function* () {
          try {
            yield chunk({ content: '<tool_call>' })
            await hold.wait
            yield chunk({ content: '{"name":"lookup","arguments":{}}</tool_call>' }, 'stop')
          } finally {
            closed()
          }
        })(),
      )
      stop.mockImplementation(async () => hold.resume())
      const onChunk = vi.fn(() => {
        if (mode === 'abort') controller.abort()
        else throw new Error('Callback failed')
      })
      const model = new BrowserChatWebLLM(page, { modelId: 'Qwen3-8B-q4f16_1-MLC', onChunk })
      const pending = model
        .bindTools([{ type: 'function', function: { name: 'lookup' } }])
        .invoke('Look up a note.', { signal: controller.signal })
      try {
        if (mode === 'abort') await expect(pending).rejects.toHaveProperty('name', 'AbortError')
        else await expect(pending).rejects.toThrow('Callback failed')
        expect(onChunk).toHaveBeenCalledTimes(1)
        expect(closed).toHaveBeenCalledTimes(1)
        expect(stop).toHaveBeenCalledTimes(1)
        expect((await plain.invoke('Try again.')).text).toBe('Recovered')
      } finally {
        hold.resume()
        await bridge.close()
      }
    },
  )

  it('does not yield a chunk after its token callback cancels', async () => {
    const { model, bridge } = browser(chunk({ content: 'Answer' }, 'stop'))
    const controller = new AbortController()
    const onToken = vi.fn(() => controller.abort())
    const callbacks = CallbackManager.fromHandlers({ handleLLMNewToken: onToken })
    const messages = [new HumanMessage('Hi')]
    const [run] = await callbacks.handleChatModelStart(model.toJSON(), [messages])
    const stream = model._streamResponseChunks(messages, { signal: controller.signal }, run)
    await expect(stream.next()).rejects.toHaveProperty('name', 'AbortError')
    expect(onToken).toHaveBeenCalledTimes(1)
    expect((await model.invoke('Next')).text).toBe('Answer')
    await bridge.close()
  })

  it('stops a native chunk batch when its callback cancels the request', async () => {
    const { page, bridge } = browser()
    const controller = new AbortController()
    const onChunk = vi.fn(() => controller.abort())
    const onToken = vi.fn()
    vi.spyOn(bridge, 'next').mockResolvedValueOnce({
      done: false,
      value: { text: 'Ignored', message: { content: 'Ignored', response_metadata: {} } },
      chunks: [chunk({ content: 'First' }), chunk({ content: 'Late' }, 'stop')],
    })
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model', onChunk })
    const output: string[] = []
    await expect(
      (async () => {
        for await (const value of await model.stream('Hi', {
          signal: controller.signal,
          callbacks: [{ handleLLMNewToken: onToken }],
        }))
          output.push(value.text)
      })(),
    ).rejects.toHaveProperty('name', 'AbortError')
    expect(onChunk).toHaveBeenCalledTimes(1)
    expect(onToken).not.toHaveBeenCalled()
    expect(output).toEqual([])
    await bridge.close()
  })

  it('delivers buffered native output before a browser stream failure', async () => {
    const { page, create, bridge } = browser(chunk({ content: 'Recovered' }, 'stop'))
    const model = new BrowserChatWebLLM(page, { modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' })
    const raw = '[{"name":"lookup","arguments":{'
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: raw })
        throw new TypeError('Native stream failed')
      })(),
    )
    const output: string[] = []
    await expect(
      (async () => {
        const stream = await model
          .bindTools([{ type: 'function', function: { name: 'lookup' } }])
          .stream('Look up A7.')
        for await (const item of stream) {
          expect(item.tool_calls ?? []).toEqual([])
          output.push(item.text)
        }
      })(),
    ).rejects.toBeInstanceOf(TypeError)
    expect(output.join('')).toBe(raw)
    expect((await model.invoke('Try again.')).text).toBe('Recovered')
    await bridge.close()
  })

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
      { role: 'tool', tool_call_id: 'picture', content: '' },
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

  it.each(['jsonMode', 'jsonSchema'])(
    'keeps the %s response format across browser sampling overrides',
    async (method) => {
      const { model, create } = browser(chunk({ content: '{"answer":42}' }, 'stop'))

      const result = await model
        .withStructuredOutput(z.object({ answer: z.number() }), { method, includeRaw: true })
        .invoke('Six times seven?', { request: { temperature: 0 } })

      expect(result.parsed).toEqual({ answer: 42 })
      const request = create.mock.calls[0]?.[0]
      expect(request).toMatchObject({
        temperature: 0,
        response_format: { type: 'json_object' },
      })
      if (method === 'jsonSchema') expect(request?.response_format?.schema).toContain('answer')
    },
  )

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

    expect(typeof evaluate.mock.calls[0]?.[1]).toBe('string')
    expect(JSON.parse(String(evaluate.mock.calls[0]?.[1]))).toMatchObject({
      messages: [{ role: 'human', content: message.content }],
    })
  })

  it('preserves literal LangChain marker keys in tool arguments', async () => {
    const { page, create } = browser(chunk({ content: 'Okay' }, 'stop'))
    page.evaluate = async (run, arg) =>
      run(
        JSON.parse(JSON.stringify(arg), (key, value) => (key === '__proto__' ? undefined : value)),
      )
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    const args = JSON.parse(
      '{"lc":1,"type":"constructor","id":["literal"],"nested":{"__lc_escaped__":{"code":"oak"}},"__proto__":{"safe":true}}',
    )
    const message = new AIMessage({
      content: [{ type: 'tool_call', id: 'call', name: 'lookup', args }],
      tool_calls: [{ id: 'call', name: 'lookup', args }],
    })

    const schema = {
      type: 'object',
      properties: JSON.parse('{"__proto__":{"type":"object"}}'),
    }
    await model
      .bindTools([{ type: 'function', function: { name: 'lookup', parameters: schema } }])
      .invoke([
        new HumanMessage('Look up the data.'),
        message,
        new ToolMessage({ tool_call_id: 'call', content: 'Done' }),
      ])

    const assistant = create.mock.calls[0]?.[0].messages[1]
    expect(assistant?.role).toBe('assistant')
    if (assistant?.role !== 'assistant') throw new Error('Missing assistant history.')
    expect(assistant.tool_calls).toHaveLength(1)
    expect(JSON.parse(assistant.tool_calls![0]!.function.arguments)).toEqual(args)
    expect(create.mock.calls[0]?.[0].tools?.[0]?.function.parameters).toEqual(schema)
  })

  it('preserves image and audio byte arrays in browser messages', async () => {
    const { page, create } = browser(chunk({ content: 'Okay' }, 'stop'))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })

    await model.invoke([
      new HumanMessage({
        content: [
          { type: 'image', data: new Uint8Array([0, 127, 255]), mimeType: 'image/png' },
          { type: 'audio', data: new Uint8Array([1, 2, 3]), mimeType: 'audio/wav' },
        ],
      }),
    ])

    expect(create.mock.calls[0]?.[0].messages[0]?.content).toEqual([
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AH//' } },
      { type: 'input_audio', input_audio: { data: 'AQID', format: 'wav' } },
    ])
  })

  it('leaves tool artifacts outside the browser request', async () => {
    const { page, create } = browser(chunk({ content: 'Okay' }, 'stop'))
    const evaluate = vi.spyOn(page, 'evaluate')
    const callback = vi.fn()
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })

    const result = await model.invoke([
      new HumanMessage('Look up the data.'),
      new AIMessage({ content: '', tool_calls: [{ id: 'call', name: 'lookup', args: {} }] }),
      new ToolMessage({ tool_call_id: 'call', content: 'Done', artifact: { callback } }),
    ])

    expect(result.text).toBe('Okay')
    expect(create).toHaveBeenCalledOnce()
    expect(callback).not.toHaveBeenCalled()
    expect(String(evaluate.mock.calls[0]?.[1])).not.toContain('artifact')
  })

  it('preserves reasoning_content when other metadata contains an lc key', async () => {
    const { model, create } = browser(chunk({ content: 'Okay' }, 'stop'))

    await model.invoke([
      new HumanMessage('Look up the data.'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call', name: 'lookup', args: {} }],
        additional_kwargs: { lc: 1, reasoning_content: 'Read the code.' },
      }),
      new ToolMessage({ tool_call_id: 'call', content: 'Done' }),
    ])

    expect(create.mock.calls[0]?.[0].messages[1]?.content).toContain(
      '<think>\nRead the code.\n</think>',
    )
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
      messages: [{ role: 'human', content: 'Hello' }],
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

  it('delivers every native chunk once when bridge reads overlap', async () => {
    const first = chunk({ content: 'Hi' })
    const last = chunk({}, 'stop')
    const { bridge } = browser(first, last)
    const id = await bridge.start({
      config: { modelId: 'test-model' },
      messages: [{ role: 'human', content: 'Hi' }],
      options: {},
      includeChunks: true,
    })
    try {
      const results = await Promise.all(Array.from({ length: 12 }, () => bridge.next(id)))
      expect(results.flatMap((result) => result.chunks ?? [])).toEqual([first, last])
      const text = results.flatMap((result) => {
        if ('error' in result) throw new Error(result.error.message)
        return result.done || 'native' in result ? [] : [result.value.text]
      })
      expect(text).toEqual(['Hi', ''])
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
    const outputTokens: number | undefined = result.usage_metadata?.output_tokens
    expect(outputTokens).toBe(3)
    expect(onChunk).toHaveBeenCalledExactlyOnceWith(native)
  })

  it('passes browser errors through and releases the failed stream', async () => {
    const { model, create, bridge } = browser(chunk({ content: 'Recovered' }, 'stop'))
    create.mockRejectedValueOnce(
      Object.assign(new Error('Context full'), { name: 'ContextWindowSizeExceededError' }),
    )
    const failed = model.invoke('Hi')
    await expect(failed).rejects.toBeInstanceOf(ContextOverflowError)
    await expect(failed).rejects.toMatchObject({
      message: 'Context full',
      cause: { name: 'ContextWindowSizeExceededError' },
    })
    expect((await model.invoke('After')).text).toBe('Recovered')
    await bridge.close()
    await expect(model.invoke('Closed')).rejects.toThrow('closed')
    expect(await bridge.next('unknown')).toMatchObject({
      error: { message: 'Unknown browser call.' },
    })
  })

  it('preserves native error causes across the browser boundary', async () => {
    const { model, create, bridge } = browser(chunk({ content: 'Recovered' }, 'stop'))
    const cause = new SyntaxError('Invalid native response')
    create.mockRejectedValueOnce(new TypeError('Native failure', { cause }))
    const failed = model.invoke('Hi')
    await expect(failed).rejects.toBeInstanceOf(TypeError)
    await expect(failed).rejects.toMatchObject({
      name: 'TypeError',
      message: 'Native failure',
      cause: expect.any(SyntaxError),
    })
    expect((await model.invoke('After')).text).toBe('Recovered')
    await bridge.close()
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

it('retries only the failed browser embedding batch', async () => {
  const { page, engine, bridge } = browser()
  const attempts: string[][] = []
  let failed = false
  vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => {
    const texts = request.input as string[]
    attempts.push([...texts])
    if (texts[0] === 'Two' && !failed) {
      failed = true
      throw new Error('Temporary failure')
    }
    return {
      object: 'list',
      model: 'embedding-model',
      usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      data: texts.map((text, index) => ({
        object: 'embedding',
        index,
        embedding: [text.length, 1],
      })),
    }
  })
  const model = new BrowserWebLLMEmbeddings(page, {
    modelId: 'embedding-model',
    batchSize: 1,
    maxRetries: 1,
  })
  expect(await model.embedDocuments(['One', 'Two', 'Three'])).toEqual([
    [3, 1],
    [3, 1],
    [5, 1],
  ])
  expect(attempts).toEqual([['One'], ['Two'], ['Two'], ['Three']])
  await bridge.close()
})

it('preserves an embedding context error across a real JSON error boundary without retrying', async () => {
  const { engine, bridge } = browser()
  const native = Object.assign(new Error('The input is too long.'), {
    name: 'EmbeddingExceedContextWindowSizeError',
  })
  const create = vi.spyOn(engine.embeddings, 'create').mockRejectedValue(native)
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const result = await Promise.resolve()
        .then(() => run(structuredClone(arg)))
        .then(
          (value) => ({ value }),
          (error: unknown) => ({
            error: {
              name: error instanceof Error ? error.name : 'Error',
              message: error instanceof Error ? error.message : String(error),
            },
          }),
        )
      if ('error' in result)
        throw Object.assign(new Error(result.error.message), { name: result.error.name })
      return JSON.parse(JSON.stringify(result.value))
    },
  }
  const model = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', maxRetries: 1 })
  const failure = await model.embedQuery('Orchid').catch((error: unknown) => error)
  expect(ContextOverflowError.isInstance(failure)).toBe(true)
  expect(getRetryable(failure)).toBe(false)
  expect(failure).toMatchObject({ cause: { name: native.name, message: native.message } })
  expect(create).toHaveBeenCalledOnce()
  await bridge.close()
})

it('does not retry a malformed browser embedding response', async () => {
  const { page, engine, bridge } = browser()
  const create = vi.spyOn(engine.embeddings, 'create').mockResolvedValue({
    object: 'list',
    model: 'embedding-model',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: [{ object: 'embedding', index: 0, embedding: [Number.NaN] }],
  })
  const model = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', maxRetries: 1 })
  const failure = await model.embedDocuments(['One']).catch((error: unknown) => error)
  expect(create).toHaveBeenCalledOnce()
  expect(failure).toBeInstanceOf(TypeError)
  await bridge.close()
})

it('checks vector dimensions across browser embedding batches', async () => {
  const { page, engine, bridge } = browser()
  let call = 0
  vi.spyOn(engine.embeddings, 'create').mockImplementation(async () => ({
    object: 'list',
    model: 'embedding-model',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: [{ object: 'embedding', index: 0, embedding: call++ === 0 ? [1, 0] : [1] }],
  }))
  const model = new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model', batchSize: 1 })
  await expect(model.embedDocuments(['One', 'Two'])).rejects.toThrow('matching dimensions')
  await bridge.close()
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

it('snapshots chat content and options before a delayed page transfer', async () => {
  const { create, bridge } = browser(chunk({ content: 'Okay' }, 'stop'))
  const hold = pause()
  const submitted = pause()
  const page: BrowserPage = {
    async evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T> {
      submitted.resume()
      await hold.wait
      return run(structuredClone(arg))
    },
  }
  const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
  const content = [{ type: 'text' as const, text: 'Before' }]
  const stop = ['Before']
  const request = { temperature: 0 }
  const stream = model._streamResponseChunks([new HumanMessage({ content })], { stop, request })
  const pending = collect(stream)
  try {
    await submitted.wait
    content[0]!.text = 'After'
    stop[0] = 'After'
    request.temperature = 1
    hold.resume()
    await pending
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      messages: [{ role: 'user', content: 'Before' }],
      stop: ['Before'],
      temperature: 0,
    })
  } finally {
    hold.resume()
    await pending.catch(() => {})
    await bridge.close()
  }
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

it('clears default sampling fields before serializing browser call overrides', async () => {
  const { page, create } = browser(chunk({ content: 'Okay' }, 'stop'))
  const model = new BrowserChatWebLLM(page, {
    modelId: 'test-model',
    maxTokens: 128,
    stop: ['END'],
  })
  const request = {}
  Object.assign(request, { max_tokens: undefined, stop: undefined })

  await model.invoke('Hi', { request })

  expect(create.mock.calls[0]?.[0].max_tokens).toBeUndefined()
  expect(create.mock.calls[0]?.[0].stop).toBeUndefined()
})

it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
  'rejects invalid browser input numbers instead of changing them to null: %s',
  async (temperature) => {
    const { model, page, create } = browser()
    const evaluate = vi.spyOn(page, 'evaluate')

    await expect(model.invoke('Hi', { request: { temperature } })).rejects.toThrow('finite numbers')

    expect(evaluate).not.toHaveBeenCalled()
    expect(create).not.toHaveBeenCalled()
  },
)
