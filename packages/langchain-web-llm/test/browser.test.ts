import { runInNewContext } from 'node:vm'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { browser } from '../fixtures/browser.js'
import { ContextOverflowError } from '@langchain/core/errors'
import { createAgent } from 'langchain'
import { AIMessage, AIMessageChunk, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { CallbackManager } from '@langchain/core/callbacks/manager'
import { z } from 'zod'
import {
  BrowserChatWebLLM,
  BrowserWebLLMEmbeddings,
  type BrowserPage,
  type WebLLMChunk,
} from '../src/index.js'
import { chunk, chunks, collect, finishRetries, pause } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

describe('browser transport', () => {
  it('checks duplicate tool arguments without losing negative zero in history', async () => {
    const { page, create, bridge } = browser(chunk({ content: 'Answer' }, 'stop'))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    const history = (value: number) => [
      new HumanMessage('Look up the value.'),
      new AIMessage({
        content: [{ type: 'tool_use', id: 'call', name: 'lookup', input: { value } }],
        tool_calls: [{ id: 'call', name: 'lookup', args: { value: -0 } }],
      }),
      new ToolMessage({ tool_call_id: 'call', content: 'Done' }),
    ]

    try {
      await expect(model.invoke(history(0))).rejects.toThrow('conflicting tool call content')
      expect(create).not.toHaveBeenCalled()
      expect((await model.invoke(history(-0))).text).toBe('Answer')
      const messages = create.mock.calls[0]![0].messages
      const assistant = messages.find((message) => message.role === 'assistant')
      const value = JSON.parse(assistant!.tool_calls![0]!.function.arguments).value
      expect(Object.is(value, -0)).toBe(true)
    } finally {
      await bridge.close()
    }
  })

  it('reads buffered responses in one browser round trip', async () => {
    const { page, model, bridge } = browser(chunk({ content: 'Answer' }, 'stop'))
    const evaluate = vi.spyOn(page, 'evaluate')

    const result = await model.invoke('Hi')

    expect(result.text).toBe('Answer')
    expect(evaluate).toHaveBeenCalledTimes(2)
    await bridge.close()
  })

  it('keeps buffered responses intact across browser batches', async () => {
    const output = Array.from({ length: 513 }, () => chunk({ content: 'x' }))
    output.push(chunk({}, 'stop'))
    const { page, model, bridge } = browser(...output)
    const evaluate = vi.spyOn(page, 'evaluate')

    try {
      const result = await model.invoke('Hi')

      expect(result.text).toBe('x'.repeat(513))
      expect(evaluate).toHaveBeenCalledTimes(3)
    } finally {
      await bridge.close()
    }
  })

  it('batches buffered reads when callback handlers ignore token events', async () => {
    const { page, model, bridge } = browser(
      chunk({ content: 'First' }),
      chunk({ content: 'Second' }),
      chunk({}, 'stop'),
    )
    const evaluate = vi.spyOn(page, 'evaluate')
    const onToken = vi.fn()
    const callbacks = CallbackManager.fromHandlers({ handleLLMNewToken: onToken })
    const [handler] = callbacks.handlers
    if (handler) handler.ignoreLLM = true

    try {
      const result = await model.invoke('Hi', { callbacks })

      expect(result.text).toBe('FirstSecond')
      expect(onToken).not.toHaveBeenCalled()
      expect(evaluate).toHaveBeenCalledTimes(2)
    } finally {
      await bridge.close()
    }
  })

  it('batches buffered reads when handlers do not receive token events', async () => {
    const { page, model, bridge } = browser(
      chunk({ content: 'First' }),
      chunk({ content: 'Second' }),
      chunk({}, 'stop'),
    )
    const evaluate = vi.spyOn(page, 'evaluate')
    const onStart = vi.fn()
    const callbacks = CallbackManager.fromHandlers({ handleLLMStart: onStart })

    try {
      const result = await model.invoke('Hi', { callbacks })

      expect(result.text).toBe('FirstSecond')
      expect(onStart).toHaveBeenCalledOnce()
      expect(evaluate).toHaveBeenCalledTimes(2)
    } finally {
      await bridge.close()
    }
  })

  it('delivers invoke callbacks before reading the next browser batch', async () => {
    const { model, create, bridge } = browser()
    const hold = pause()
    const onToken = vi.fn()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'First' })
        await hold.wait
        yield chunk({ content: 'Second' }, 'stop')
      })(),
    )
    try {
      const pending = model.invoke('Hi', { callbacks: [{ handleLLMNewToken: onToken }] })
      await vi.waitFor(() => expect(onToken).toHaveBeenCalledOnce())
      expect(onToken.mock.calls[0]?.[0]).toBe('First')
      hold.resume()
      expect((await pending).text).toBe('FirstSecond')
    } finally {
      hold.resume()
      await bridge.close()
    }
  })

  it('cancels while the browser start response is pending', async () => {
    const { page, bridge, create } = browser()
    const started = pause()
    const release = pause()
    const stop = vi.spyOn(bridge, 'stop')
    let first = true
    const delayed: BrowserPage = {
      async evaluate(run, arg) {
        const value = await page.evaluate(run, arg)
        if (first) {
          first = false
          started.resume()
          await release.wait
        }
        return value
      },
    }
    const model = new BrowserChatWebLLM(delayed, { modelId: 'test-model' })
    const controller = new AbortController()
    const pending = model
      ._streamResponseChunks([new HumanMessage('Hi')], { signal: controller.signal })
      .next()
    const failed = vi.fn()
    void pending.catch(failed)
    try {
      await started.wait
      controller.abort()
      await vi.waitFor(() => expect(failed).toHaveBeenCalledWith(controller.signal.reason))
      release.resume()
      await vi.waitFor(() => expect(stop).toHaveBeenCalledOnce())
      expect(create).not.toHaveBeenCalled()
    } finally {
      release.resume()
      await pending.catch(() => {})
      await bridge.close()
    }
  })

  it('cancels while a browser read response is pending', async () => {
    const { page, bridge } = browser()
    const reading = pause()
    const release = pause()
    const stop = vi.spyOn(bridge, 'stop')
    let count = 0
    const delayed: BrowserPage = {
      async evaluate(run, arg) {
        const value = await page.evaluate(run, arg)
        if (++count === 2) {
          reading.resume()
          await release.wait
        }
        return value
      },
    }
    const model = new BrowserChatWebLLM(delayed, { modelId: 'test-model' })
    const controller = new AbortController()
    const pending = model
      ._streamResponseChunks([new HumanMessage('Hi')], { signal: controller.signal })
      .next()
    const failed = vi.fn()
    void pending.catch(failed)
    try {
      await reading.wait
      controller.abort()
      await vi.waitFor(() => expect(failed).toHaveBeenCalledWith(controller.signal.reason))
      expect(stop).toHaveBeenCalledOnce()
    } finally {
      release.resume()
      await pending.catch(() => {})
      await bridge.close()
    }
  })

  it('aborts a buffered browser read while it waits for more output', async () => {
    const { model, create, bridge } = browser()
    const stop = vi.spyOn(bridge, 'stop')
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'First' })
        await hold.wait
        yield chunk({ content: 'Second' }, 'stop')
      })(),
    )
    const controller = new AbortController()
    const pending = model.invoke('Hi', { signal: controller.signal })
    try {
      await vi.waitFor(() => expect(create).toHaveBeenCalledOnce())
      const cancelled = expect(pending).rejects.toHaveProperty('name', 'AbortError')
      controller.abort()
      await vi.waitFor(() => expect(stop).toHaveBeenCalledOnce())
      hold.resume()
      await cancelled
    } finally {
      hold.resume()
      await bridge.close()
    }
  })

  it('returns cancellation without waiting for the browser to stop', async () => {
    const { model, bridge, create } = browser()
    const reading = pause()
    const release = pause()
    const stopping = pause()
    const releaseStop = pause()
    const stop = bridge.stop.bind(bridge)
    vi.spyOn(bridge, 'stop').mockImplementation(async (id) => {
      stopping.resume()
      await releaseStop.wait
      await stop(id)
    })
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          reading.resume()
          await release.wait
          return { done: true as const, value: undefined }
        },
        return: async () => ({ done: true as const, value: undefined }),
      }),
    })
    const controller = new AbortController()
    const pending = model.invoke('Hi', { signal: controller.signal })
    const failed = vi.fn()
    void pending.catch(failed)
    try {
      await reading.wait
      controller.abort()
      await stopping.wait
      await vi.waitFor(() => expect(failed).toHaveBeenCalledWith(controller.signal.reason))
    } finally {
      release.resume()
      releaseStop.resume()
      await pending.catch(() => {})
      await bridge.close()
    }
  })

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

    if (retries)
      expect((await finishRetries(() => model.invoke('Hi', { maxRetries: retries }))).text).toBe(
        'Okay',
      )
    else await expect(model.invoke('Hi', { maxRetries: retries })).rejects.toThrow('Opening failed')
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

  it.each(['lc_prefer_streaming', 'lc_prefer_chat_model_stream_events'] as const)(
    'does not run a truncated tool call with %s across the browser boundary',
    async (preference) => {
      const { model, create, bridge } = browser()
      const complete = '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
      const partial = '<tool_call>{"name":"lookup","arguments":{"code":"A8'
      create.mockResolvedValueOnce(
        chunks(chunk({ content: 'Checking.' + complete + partial }), chunk({}, 'length')),
      )
      create.mockResolvedValueOnce(chunks(chunk({ content: 'Done.' }, 'stop')))
      const execute = vi.fn(() => 'Done')
      const lookup = tool(execute, { name: 'lookup', schema: z.object({ code: z.string() }) })
      const callbacks = CallbackManager.fromHandlers({
        handleLLMNewToken: () => {},
        handleChatModelStreamEvent: () => {},
      })
      const [handler] = callbacks.handlers
      if (!handler) throw new Error('Missing callback handler.')
      Object.assign(handler, { [preference]: true })

      try {
        const result = await createAgent({ model, tools: [lookup] }).invoke(
          { messages: [new HumanMessage('Look up codes A7 and A8.')] },
          { callbacks },
        )
        expect(execute).not.toHaveBeenCalled()
        expect(result.messages.at(-1)?.text).toBe('Checking.')
        expect(result.messages.at(-1)?.response_metadata.finish_reason).toBe('length')
        expect(
          (await model.invoke([...result.messages, new HumanMessage('Try again.')])).text,
        ).toBe('Done.')
      } finally {
        await bridge.close()
      }
    },
  )

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

  it.each(['invoke', 'stream'] as const)(
    'preserves reserved keys in tool calls returned from the browser: %s',
    async (method) => {
      const input = '{"__proto__":{"safe":true},"code":"A7"}'
      const content = `<tool_call>{"name":"lookup","arguments":${input}}</tool_call>`
      const { page, bridge } = browser(chunk({ content }, 'stop'))
      page.evaluate = async (run, arg) => {
        const result = await run(arg)
        return result === undefined
          ? result
          : JSON.parse(JSON.stringify(result), (key, value) =>
              key === '__proto__' ? undefined : value,
            )
      }
      const model = new BrowserChatWebLLM(page, { modelId: 'Qwen3-8B-q4f16_1-MLC' })
      try {
        const bound = model.bindTools([
          { type: 'function', function: { name: 'lookup', parameters: { type: 'object' } } },
        ])
        const calls =
          method === 'invoke'
            ? ((await bound.invoke('Look up the code.')).tool_calls ?? [])
            : (await collect(await bound.stream('Look up the code.'))).flatMap(
                (message) => message.tool_calls ?? [],
              )
        const call = calls.find((call) => call.id)
        expect(JSON.stringify(call?.args)).toBe(input)
      } finally {
        await bridge.close()
      }
    },
  )

  it('preserves negative zero in streamed native tool arguments', async () => {
    const { page, bridge } = browser(
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{"__proto__":{"values":[-0,0]},"value":-0}' },
            },
          ],
        },
        'tool_calls',
      ),
    )
    page.evaluate = async (run, arg) => {
      const result = await run(arg)
      return result === undefined
        ? result
        : JSON.parse(JSON.stringify(result), (key, value) =>
            key === '__proto__' ? undefined : value,
          )
    }
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    try {
      const tool = {
        type: 'function' as const,
        function: { name: 'lookup', parameters: { type: 'object' } },
      }
      const stream = await model.bindTools([tool]).stream('Look up the code.')
      const calls = (await collect(stream)).flatMap((message) => message.tool_calls ?? [])
      const call = calls.find((call) => call.id)
      expect(Object.is(call?.args.value, -0)).toBe(true)
      const nested = Object.getOwnPropertyDescriptor(call?.args ?? {}, '__proto__')?.value
      expect(Object.is(nested?.values[0], -0)).toBe(true)
      expect(Object.is(nested?.values[1], 0)).toBe(true)
    } finally {
      await bridge.close()
    }
  })

  it('preserves Float32 PCM samples and sample rate through JSON browser transport', async () => {
    const { page, create, bridge } = browser(chunk({ content: 'Heard it' }, 'stop'))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    const samples = new Float32Array([9, -0, 0.25, -0.5, 9]).subarray(1, 4)
    const before = [...samples]
    try {
      await model.invoke([
        new HumanMessage({
          content: [
            { type: 'text', text: 'Listen.' },
            {
              type: 'input_audio',
              input_audio: { format: 'pcm_f32', data: samples, sample_rate: 24000 },
            },
          ],
        }),
      ])
      const content = create.mock.calls[0]![0].messages[0]!.content
      if (!Array.isArray(content) || content[1]?.type !== 'input_audio')
        throw new Error('Missing audio input.')
      const audio = content[1].input_audio
      if (audio.format !== 'wav') throw new Error('Expected transport-safe WAV bytes.')
      const bytes = Buffer.from(audio.data, 'base64')
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
      expect(view.getUint16(20, true)).toBe(3)
      expect(view.getUint32(24, true)).toBe(24000)
      expect(view.getUint32(52, true)).toBe(samples.byteLength)
      expect(
        before.every((sample, index) => Object.is(view.getFloat32(56 + index * 4, true), sample)),
      ).toBe(true)
      expect([...samples]).toEqual(before)
    } finally {
      await bridge.close()
    }
  })

  it.each([0, 16000])(
    'rejects detached PCM views before browser calls or sample-rate checks (%i)',
    async (sampleRate) => {
      const { page, create, bridge } = browser()
      const evaluate = vi.spyOn(page, 'evaluate')
      const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
      const samples = new Float32Array([0.25])
      structuredClone(samples.buffer, { transfer: [samples.buffer] })
      try {
        await expect(
          model.invoke([
            new HumanMessage({
              content: [
                {
                  type: 'input_audio',
                  input_audio: { format: 'pcm_f32', data: samples, sample_rate: sampleRate },
                },
              ],
            }),
          ]),
        ).rejects.toBeInstanceOf(TypeError)
        expect(evaluate).not.toHaveBeenCalled()
        expect(create).not.toHaveBeenCalled()
      } finally {
        await bridge.close()
      }
    },
  )

  it('rejects PCM above WebLLM’s sample-rate limit before browser calls', async () => {
    const { page, create, bridge } = browser()
    const evaluate = vi.spyOn(page, 'evaluate')
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    try {
      await expect(
        model.invoke([
          new HumanMessage({
            content: [
              {
                type: 'input_audio',
                input_audio: {
                  format: 'pcm_f32',
                  data: new Float32Array([0.25]),
                  sample_rate: 384_001,
                },
              },
            ],
          }),
        ]),
      ).rejects.toThrow('384000')
      expect(evaluate).not.toHaveBeenCalled()
      expect(create).not.toHaveBeenCalled()
    } finally {
      await bridge.close()
    }
  })

  it.each(['data:', 'DATA:'])(
    'preserves inline WAV audio with %s in browser messages',
    async (scheme) => {
      const { page, create, bridge } = browser(chunk({ content: 'Heard it' }, 'stop'))
      const url = scheme + 'audio/wav;base64,AQ=='
      const model = new BrowserChatWebLLM(page, { modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })
      const message = new HumanMessage({ content: [{ type: 'audio', url }] })
      try {
        await model.invoke([message])
        expect(create.mock.calls[0]?.[0].messages[0]?.content).toEqual([
          {
            type: 'input_audio',
            input_audio: { format: 'wav', data: 'data:audio/wav;base64,AQ==' },
          },
        ])
        expect(message.content).toEqual([{ type: 'audio', url }])
      } finally {
        await bridge.close()
      }
    },
  )

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

  it('rejects transferred image bytes before calling the browser', async () => {
    const { page, create } = browser(chunk({ content: 'Okay' }, 'stop'))
    const evaluate = vi.spyOn(page, 'evaluate')
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    const bytes = new Uint8Array([1, 2, 3])
    structuredClone(bytes, { transfer: [bytes.buffer] })

    await expect(
      model.invoke([
        new HumanMessage({ content: [{ type: 'image', data: bytes, mimeType: 'image/png' }] }),
      ]),
    ).rejects.toThrow(TypeError)
    expect(evaluate).not.toHaveBeenCalled()
    expect(create).not.toHaveBeenCalled()
  })

  it('preserves image and audio bytes from another realm in browser messages', async () => {
    const { page, create, bridge } = browser(chunk({ content: 'Okay' }, 'stop'))
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model' })
    const image = runInNewContext('new Uint8Array([0, 127, 255])') as Uint8Array
    const audio = runInNewContext('new Uint8Array([1, 2, 3])') as Uint8Array

    try {
      await model.invoke([
        new HumanMessage({
          content: [
            { type: 'image', data: image, mimeType: 'image/png' },
            { type: 'audio', data: audio, mimeType: 'audio/wav' },
          ],
        }),
      ])

      expect(create.mock.calls[0]?.[0].messages[0]?.content).toEqual([
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AH//' } },
        { type: 'input_audio', input_audio: { data: 'AQID', format: 'wav' } },
      ])
    } finally {
      await bridge.close()
    }
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

  it('keeps browser output intact when the host callback mutates native chunks', async () => {
    const { page, bridge, create } = browser()
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Answer' }, 'stop')))
    const onChunk = vi.fn((native: WebLLMChunk) => {
      const choice = native.choices[0]
      if (choice && 'delta' in choice) choice.delta.content = 'Changed'
    })
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model', onChunk })

    try {
      const result = await model.invoke('Hi')
      expect(onChunk).toHaveBeenCalledOnce()
      expect(result.text).toBe('Answer')
    } finally {
      await bridge.close()
    }
  })

  it('does not copy native chunks for the internal browser relay', async () => {
    const { bridge, create } = browser()
    const native = chunk({ content: 'Hi' }, 'stop')
    const copyChoices = vi.spyOn(native.choices, 'map')
    create.mockResolvedValueOnce(chunks(native))
    const id = await bridge.start({
      config: { modelId: 'test-model' },
      messages: [{ role: 'human', content: 'Hi' }],
      options: {},
      includeChunks: true,
    })
    const received = []
    try {
      while (true) {
        const result = await bridge.next(id)
        if ('error' in result) throw new Error(result.error.message)
        received.push(...(result.chunks ?? []))
        if (result.done) break
      }
      expect(received[0]).toBe(native)
      expect(copyChoices).not.toHaveBeenCalled()
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
    create.mockResolvedValueOnce(
      chunks({ ...chunk({ content: '<think>Done.</think>42' }, 'stop'), id: 'answer-response' }),
    )
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

  it('preserves non-JSON log probabilities across browser evaluation', async () => {
    const native = chunk({ content: 'Okay' }, 'stop')
    native.choices[0]!.logprobs = {
      content: [
        {
          token: 'Okay',
          bytes: [79, 107, 97, 121],
          logprob: Number.NEGATIVE_INFINITY,
          top_logprobs: [],
        },
      ],
    }
    const { model, bridge } = browser(native)

    try {
      const result = await model.invoke('Hi')

      expect(result.response_metadata.logprobs).toEqual({
        content: [
          {
            token: 'Okay',
            bytes: [79, 107, 97, 121],
            logprob: Number.NEGATIVE_INFINITY,
            top_logprobs: [],
          },
        ],
      })
    } finally {
      await bridge.close()
    }
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

  it('waits for a native call and rejects queued work when closing', async () => {
    const { bridge } = browser()
    const entered = pause()
    const finish = pause()
    const active = bridge.withEngine(async () => {
      entered.resume()
      await finish.wait
      return 'finished'
    })
    await entered.wait
    const run = vi.fn()
    const queued = expect(bridge.withEngine(run)).rejects.toThrow('closed')
    let closed = false
    const closing = bridge.close().then(() => {
      closed = true
    })
    await expect(bridge.withEngine(run)).rejects.toThrow('closed')
    expect(closed).toBe(false)
    finish.resume()
    expect(await active).toBe('finished')
    await queued
    await closing
    expect(closed).toBe(true)
    expect(run).not.toHaveBeenCalled()
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
    await expect(collect(stream)).rejects.toHaveProperty('name', 'AbortError')
    expect((await model.invoke('After')).text).toBe('Recovered')
  })
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
