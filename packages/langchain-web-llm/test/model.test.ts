import { describe, expect, it, vi } from 'vitest'
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { CallbackManager } from '@langchain/core/callbacks/manager'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { RunnableLambda } from '@langchain/core/runnables'
import { createAgent } from 'langchain'
import { createDeepAgent, createUnsupportedContentMiddleware } from 'deepagents'
import { z } from 'zod'
import { ChatWebLLM } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

const lookup = tool(({ code }) => `Result ${code}: 42`, {
  name: 'lookup',
  description: 'Look up the answer for a code.',
  schema: z.object({ code: z.string() }),
})
const toolOutput =
  '<tool_call>\n<function=lookup>\n<parameter=code>\nA7\n</parameter>\n</function>\n</tool_call>'
const secondToolOutput =
  '<tool_call>\n<function=lookup>\n<parameter=code>\nA8\n</parameter>\n</function>\n</tool_call>'
const qwen = 'Qwen3.5-0.8B-q4f16_1-MLC'

function qwenModel(...items: ReturnType<typeof chunk>[]) {
  const result = setup(...items)
  return {
    ...result,
    model: new ChatWebLLM({
      engine: result.engine,
      modelId: qwen,
      request: { extra_body: { enable_thinking: true } },
    }),
  }
}

describe('LangChain model', () => {
  it('forwards content stream events to handlers that request them', async () => {
    const { model } = setup(chunk({ content: 'Answer' }, 'stop'))
    const onEvent = vi.fn()
    const callbacks = CallbackManager.fromHandlers({ handleChatModelStreamEvent: onEvent })
    const [handler] = callbacks.handlers
    if (!handler) throw new Error('Missing callback handler.')
    Object.assign(handler, { lc_prefer_chat_model_stream_events: true })

    const result = await model.invoke('Hi', { callbacks })

    expect(result.text).toBe('Answer')
    expect(onEvent.mock.calls.map(([event]) => event.event)).toEqual(
      expect.arrayContaining(['message-start', 'content-block-delta', 'message-finish']),
    )
  })

  it.each(['message-start', 'content-block-start', 'content-block-finish'])(
    'stops content events when its consumer cancels at %s',
    async (phase) => {
      const { model } = setup(chunk({ content: 'Answer' }, 'stop'))
      const controller = new AbortController()
      const after: ChatModelStreamEvent[] = []
      await expect(
        (async () => {
          for await (const event of model._streamChatModelEvents([new HumanMessage('Hi')], {
            signal: controller.signal,
          })) {
            if (controller.signal.aborted) after.push(event)
            if (event.event === phase) controller.abort()
          }
        })(),
      ).rejects.toHaveProperty('name', 'AbortError')
      expect(after).toEqual([])
      expect((await model.invoke('Next')).text).toBe('Answer')
    },
  )

  it('does not yield a chunk after its token callback cancels', async () => {
    const { model } = setup(chunk({ content: 'Answer' }, 'stop'))
    const controller = new AbortController()
    const onToken = vi.fn(() => controller.abort())
    const callbacks = CallbackManager.fromHandlers({ handleLLMNewToken: onToken })
    const messages = [new HumanMessage('Hi')]
    const [run] = await callbacks.handleChatModelStart(model.toJSON(), [messages])
    const stream = model._streamResponseChunks(messages, { signal: controller.signal }, run)
    await expect(stream.next()).rejects.toHaveProperty('name', 'AbortError')
    expect(onToken).toHaveBeenCalledTimes(1)
    expect((await model.invoke('Next')).text).toBe('Answer')
  })

  it('skips token dispatch for handlers that ignore LLM events', async () => {
    const { model } = setup(chunk({ content: 'Answer' }, 'stop'))
    const onToken = vi.fn()
    const callbacks = CallbackManager.fromHandlers({ handleLLMNewToken: onToken })
    const [handler] = callbacks.handlers
    if (handler) handler.ignoreLLM = true
    const messages = [new HumanMessage('Hi')]
    const [run] = await callbacks.handleChatModelStart(model.toJSON(), [messages])
    if (!run) throw new Error('Missing callback run manager.')
    const dispatch = vi.spyOn(run, 'handleLLMNewToken')

    const output = await collect(model._streamResponseChunks(messages, {}, run))

    expect(output.map((item) => item.text).join('')).toBe('Answer')
    expect(dispatch).not.toHaveBeenCalled()
    expect(onToken).not.toHaveBeenCalled()
  })

  it('stops pending tool chunks when a consumer cancels', async () => {
    const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            tool_calls: [
              { index: 0, function: { name: 'lookup', arguments: '{"code":"A7"}' } },
              { index: 1, function: { name: 'lookup', arguments: '{"code":"A8"}' } },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    const controller = new AbortController()
    const pending = model._streamResponseChunks([new HumanMessage('Find both codes.')], {
      tools: [lookup],
      signal: controller.signal,
    })
    expect((await pending.next()).value?.message).toMatchObject({ tool_calls: [] })
    controller.abort()
    await expect(pending.next()).rejects.toHaveProperty('name', 'AbortError')
    expect((await model.invoke('Next')).text).toBe('Recovered')
  })

  it('preserves native tool JSON with finite numbers and literal protocol tags', async () => {
    const input = ' { "small": 1e-300, "large": 1e300, "text": "</tool_call>", "items": [null] } '
    const { model } = setup(
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: input } }] },
        'tool_calls',
      ),
    )
    const result = await model.bindTools([lookup]).invoke('Look up the code.')
    expect(result.tool_call_chunks?.[0]?.args).toBe(input)
    expect(result.tool_calls?.[0]?.args).toEqual(JSON.parse(input))
  })

  it('preserves assistant text streamed alongside a native tool call', async () => {
    const { model } = setup(
      chunk(
        {
          content: 'I will check that.',
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
          ],
        },
        'tool_calls',
      ),
    )

    const result = await model.bindTools([lookup]).invoke('Look up A7.')

    expect(result.text).toBe('I will check that.')
    expect(result.tool_calls).toMatchObject([
      { id: 'call-1', name: 'lookup', args: { code: 'A7' } },
    ])
  })

  it.each(['{"code":1e999}', '{"code":{"values":[-1e999]}}'])(
    'rejects overflowing native tool arguments before running a tool: %s',
    async (input) => {
      const { model, create } = setup()
      create.mockResolvedValueOnce(
        chunks(
          chunk(
            { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: input } }] },
            'tool_calls',
          ),
        ),
      )
      create.mockResolvedValueOnce(chunks(chunk({ content: 'Done' }, 'stop')))
      const callback = vi.fn(() => 'Done')
      const lookup = tool(callback, { name: 'lookup', schema: z.object({ code: z.unknown() }) })
      const agent = createAgent({ model, tools: [lookup] })
      await expect(
        agent.invoke({ messages: [new HumanMessage('Look up the code.')] }),
      ).rejects.toThrow('finite numbers')
      expect(callback).not.toHaveBeenCalled()
      expect(create).toHaveBeenCalledTimes(1)
      expect((await model.invoke('Next')).text).toBe('Done')
    },
  )

  it('does not run a native tool when its arguments are malformed', async () => {
    const { model, create } = setup()
    create.mockResolvedValueOnce(
      chunks(
        chunk({
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{' },
            },
          ],
        }),
        chunk(
          { tool_calls: [{ index: 0, type: 'function', function: { arguments: 'invalid' } }] },
          'tool_calls',
        ),
      ),
    )
    const callback = vi.fn(() => 'Done')
    const lookup = tool(callback, {
      name: 'lookup',
      description: 'Look up the answer for a code.',
      schema: z.object({ code: z.unknown() }),
    })
    const agent = createAgent({ model, tools: [lookup] })

    await expect(
      agent.invoke({ messages: [new HumanMessage('Look up the code.')] }),
    ).rejects.toThrow()
    expect(callback).not.toHaveBeenCalled()
    expect(create).toHaveBeenCalledTimes(1)
  })

  it('does not run a valid Qwen tool before a later invalid call fails the stream', async () => {
    const missingToolOutput = toolOutput.replace('<function=lookup>', '<function=missing>')
    const { model, create } = qwenModel(
      chunk({ content: toolOutput }),
      chunk({ content: missingToolOutput }, 'stop'),
    )
    const callback = vi.fn(() => 'Done')
    const agent = createAgent({
      model,
      tools: [
        tool(callback, {
          name: 'lookup',
          description: 'Look up the answer for a code.',
          schema: z.object({ code: z.string() }),
        }),
      ],
    })

    await expect(
      agent.invoke({ messages: [new HumanMessage('Look up the code.')] }),
    ).rejects.toThrow(/unknown tool/i)
    expect(callback).not.toHaveBeenCalled()
    expect(create).toHaveBeenCalledTimes(1)
  })

  it('can cancel from a reasoning callback and use the engine again', async () => {
    const { model, create, stop } = qwenModel(chunk({ content: 'Recovered' }, 'stop'))
    create.mockResolvedValueOnce(
      chunks(
        chunk({ content: '<think>Planning.' }),
        chunk({ content: '</think>Too late.' }, 'stop'),
      ),
    )
    const cancel = new AbortController()
    await expect(
      model.invoke('Hi', {
        signal: cancel.signal,
        callbacks: [
          {
            handleLLMNewToken(...args) {
              const chunk = args[5]?.chunk
              if (
                chunk &&
                'message' in chunk &&
                chunk.message.contentBlocks.some((block) => block.type === 'reasoning')
              )
                cancel.abort()
            },
          },
        ],
      }),
    ).rejects.toHaveProperty('name', 'AbortError')
    expect(stop).toHaveBeenCalledTimes(1)
    expect((await model.invoke('Next')).text).toBe('Recovered')
  })

  it('forwards reasoning, tool calls and final metadata through callbacks', async () => {
    const { model } = qwenModel(
      chunk({ content: '<think>Need the lookup.</think>' + toolOutput }, 'stop'),
    )
    const seen: AIMessageChunk[] = []
    await model.bindTools([lookup]).invoke('Look up A7.', {
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
    expect(
      seen.flatMap((message) => message.tool_call_chunks ?? []).map((call) => call.index),
    ).toEqual([0, 0])
    expect(seen.at(-1)?.response_metadata.finish_reason).toBe('tool_calls')
  })

  it('streams reasoning and tools from a nested invoke through standard events', async () => {
    const { model } = qwenModel(
      chunk({ content: '<think>Need the lookup.</think>' + toolOutput }, 'stop'),
    )
    const run = RunnableLambda.from((input: string, config) =>
      model.bindTools([lookup]).invoke(input, config),
    )
    const events = await collect(run.streamEvents('Look up A7.', { version: 'v2' }))
    const messages = events
      .filter((event) => event.event === 'on_chat_model_stream')
      .map((event) => event.data.chunk as AIMessageChunk)
    expect(messages.flatMap((message) => message.contentBlocks)).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'Need the lookup.' }),
    )
    expect(messages.flatMap((message) => message.tool_calls ?? [])).toMatchObject([
      { name: 'lookup', args: { code: 'A7' } },
    ])
  })

  it('streams reasoning, tool calls, and usage through LangChain content streams', async () => {
    const usage = {
      prompt_tokens: 3,
      completion_tokens: 2,
      total_tokens: 5,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 2,
        decode_tokens_per_s: 3,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.2,
      },
    }
    const response = {
      ...chunk(
        {
          content:
            '<think>Need the lookup.</think>Checking both codes.' + toolOutput + secondToolOutput,
        },
        'stop',
      ),
      usage,
    }
    const { model } = qwenModel(response)
    const stream = model.streamEvents('Look up A7.', { tools: [lookup] })

    const [reasoning, text, toolCalls, tokenUsage, message] = await Promise.all([
      stream.reasoning,
      stream.text,
      stream.toolCalls,
      stream.usage,
      stream,
    ])

    expect(reasoning).toContain('Need the lookup.')
    expect(text).toContain('Checking both codes.')
    expect(toolCalls).toMatchObject([
      { name: 'lookup', args: { code: 'A7' } },
      { name: 'lookup', args: { code: 'A8' } },
    ])
    expect(tokenUsage).toMatchObject({ input_tokens: 3, output_tokens: 2, total_tokens: 5 })
    expect(message.tool_calls).toMatchObject([
      { name: 'lookup', args: { code: 'A7' } },
      { name: 'lookup', args: { code: 'A8' } },
    ])

    const events = await collect(stream)
    expect(events).toContainEqual({
      event: 'content-block-start',
      index: 1,
      content: { type: 'text', text: '' },
    })
    expect(events).toContainEqual({
      event: 'content-block-delta',
      index: 1,
      delta: { type: 'text-delta', text: 'Checking both codes.' },
    })
    expect(events).toContainEqual({
      event: 'message-finish',
      reason: 'tool_use',
      usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
      responseMetadata: expect.objectContaining({ model_provider: 'webllm' }),
    })
    const blockIndexes = events
      .filter((event) => event.event === 'content-block-start')
      .map((event) => event.index)
    expect(new Set(blockIndexes).size).toBe(blockIndexes.length)
  })

  it('keeps sparse content indices separate from tool calls without changing output', async () => {
    const { model } = setup()
    const messages = [
      new AIMessageChunk({
        content: [
          { type: 'reasoning', reasoning: 'Thought', index: 3 },
          { type: 'text', text: '  Answer', index: 1 },
        ],
      }),
      new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          { index: 8, id: 'first', name: 'lookup', args: '{"code":"A7"}' },
          { index: 9, id: 'second', name: 'lookup', args: '{"code":"A8"}' },
        ],
      }),
      new AIMessageChunk({ content: ' tail\n' }),
      new AIMessageChunk({ content: [{ type: 'reasoning', reasoning: 'After' }] }),
      new AIMessageChunk({ content: '', response_metadata: { finish_reason: 'tool_calls' } }),
    ]
    vi.spyOn(model, '_streamResponseChunks').mockImplementation(async function* () {
      for (const message of messages) yield new ChatGenerationChunk({ text: '', message })
    })

    const stream = model.streamEvents('Hi')
    const events = await collect(stream)
    const result = await stream

    expect(
      events.filter((event) => event.event === 'content-block-start').map((event) => event.index),
    ).toEqual([3, 1, 0, 2, 4])
    expect(events.filter((event) => event.event === 'content-block-finish')).toEqual([
      {
        event: 'content-block-finish',
        index: 3,
        content: { type: 'reasoning', reasoning: 'Thought' },
      },
      {
        event: 'content-block-finish',
        index: 1,
        content: { type: 'text', text: '  Answer tail\n' },
      },
      {
        event: 'content-block-finish',
        index: 0,
        content: { type: 'tool_call', id: 'first', name: 'lookup', args: { code: 'A7' } },
      },
      {
        event: 'content-block-finish',
        index: 2,
        content: { type: 'tool_call', id: 'second', name: 'lookup', args: { code: 'A8' } },
      },
      {
        event: 'content-block-finish',
        index: 4,
        content: { type: 'reasoning', reasoning: 'After' },
      },
    ])
    expect(result.text).toBe('  Answer tail\n')
    expect(result.tool_calls).toMatchObject([
      { id: 'first', name: 'lookup', args: { code: 'A7' } },
      { id: 'second', name: 'lookup', args: { code: 'A8' } },
    ])
    expect(events.at(-1)).toMatchObject({ event: 'message-finish', reason: 'tool_use' })
  })

  it.each([
    ['stop', 'stop'],
    ['length', 'length'],
  ] as const)(
    'maps WebLLM %s endings in LangChain content streams',
    async (webllmReason, reason) => {
      const { model } = setup(chunk({ content: 'Answer' }, webllmReason))
      const events = await collect(model.streamEvents('Hi'))
      expect(events.at(-1)).toMatchObject({ event: 'message-finish', reason })
      const text = events
        .flatMap((event) =>
          event.event === 'content-block-delta' && event.delta.type === 'text-delta'
            ? [event.delta.text]
            : [],
        )
        .join('')
      expect(text).toBe('Answer')
    },
  )

  it('streams tool calls through LangChain Core accessors', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [{ index: 0, function: { name: 'lookup', arguments: '{"code":"A7"}' } }],
        },
        'tool_calls',
      ),
    )
    const stream = model.streamEvents('Look up A7', { tools: [lookup] })

    expect(await collect(stream.toolCalls)).toMatchObject([
      { name: 'lookup', args: { code: 'A7' } },
    ])
  })

  it('streams tool arguments with LangChain Core block deltas', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
          ],
        },
        'tool_calls',
      ),
    )
    const stream = model.streamEvents('Look up A7', { tools: [lookup] })
    const [events, toolCalls] = await Promise.all([collect(stream), stream.toolCalls])
    const start = events.find(
      (event) => event.event === 'content-block-start' && event.content.type === 'tool_call_chunk',
    )
    expect(start).toMatchObject({
      event: 'content-block-start',
      content: { type: 'tool_call_chunk', name: 'lookup' },
    })
    if (start?.event !== 'content-block-start') throw new Error('Missing tool-call start event.')
    expect(events).toContainEqual({
      event: 'content-block-delta',
      index: start.index,
      delta: { type: 'block-delta', fields: { type: 'tool_call_chunk', args: '{"code":"A7"}' } },
    })
    expect(events).toContainEqual({
      event: 'content-block-finish',
      index: start.index,
      content: { type: 'tool_call', id: 'call-1', name: 'lookup', args: { code: 'A7' } },
    })
    expect(toolCalls).toMatchObject([{ id: 'call-1', name: 'lookup', args: { code: 'A7' } }])
  })

  it('returns text, usage, and native metadata without owning the engine', async () => {
    const usage = {
      prompt_tokens: 3,
      completion_tokens: 2,
      total_tokens: 5,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 2,
        decode_tokens_per_s: 3,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.2,
      },
    }
    const { model, engine, create } = setup(chunk({ content: 'Hello' }, 'stop'), {
      ...chunk(),
      usage,
    })
    const unload = vi.spyOn(engine, 'unload')
    const result = await model.invoke('Hi')
    expect(result.text).toBe('Hello')
    expect(result.usage_metadata).toMatchObject({
      input_tokens: 3,
      output_tokens: 2,
      total_tokens: 5,
    })
    const outputTokens: number | undefined = result.usage_metadata?.output_tokens
    expect(outputTokens).toBe(2)
    expect(result.response_metadata).toMatchObject({ finish_reason: 'stop', webllm: usage.extra })
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [{ role: 'user', content: 'Hi' }],
        n: 1,
        stream: true,
        stream_options: { include_usage: true },
      }),
    )
    expect(unload).not.toHaveBeenCalled()
  })

  it('streams the first token before generation finishes and runs callbacks', async () => {
    const { model, create } = setup()
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'A' })
        await hold.wait
        yield chunk({ content: 'B' }, 'stop')
      })(),
    )
    const callback = vi.fn()
    const stream = await model.stream('Hi', { callbacks: [{ handleLLMNewToken: callback }] })
    const first = await stream.next()
    expect(first.value?.text).toBe('A')
    hold.resume()
    await collect(stream)
    expect(callback.mock.calls.map(([token]) => token).join('')).toBe('AB')
  })

  it('runs the callback before yielding when the consumer stops early', async () => {
    const { model } = setup(chunk({ content: 'A' }), chunk({ content: 'B' }, 'stop'))
    const callback = vi.fn()
    const stream = await model.stream('Hi', { callbacks: [{ handleLLMNewToken: callback }] })

    expect((await stream.next()).value?.text).toBe('A')
    expect(callback).toHaveBeenCalledOnce()
    await stream.return()
  })

  it('works in a prompt chain, batch, and message serialization', async () => {
    const { model } = setup(chunk({ content: 'Hello' }, 'stop'))
    const chain = ChatPromptTemplate.fromMessages([
      ['system', 'Be concise'],
      ['human', 'Hi {name}'],
    ]).pipe(model)
    expect((await chain.invoke({ name: 'Ada' })).text).toBe('Hello')
    expect((await model.batch(['One', 'Two'])).map((message) => message.text)).toEqual([
      'Hello',
      'Hello',
    ])
    expect(JSON.stringify(model.toJSON())).not.toContain('engine')
  })

  it('passes native options and per-call settings without mutating them', async () => {
    const { engine, create } = setup(chunk({ content: 'yes' }, 'stop'))
    const request = { temperature: 0.5, top_p: 0.8, extra_body: { enable_thinking: false } }
    const model = new ChatWebLLM({ engine, modelId: 'test-model', request })
    request.temperature = 9
    await model.invoke('Hi', { request: { temperature: 0.2, logprobs: true }, stop: ['END'] })
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0.2, top_p: 0.8, logprobs: true, stop: ['END'] }),
    )
    expect(model.getLsParams({})).toMatchObject({
      ls_provider: 'webllm',
      ls_model_name: 'test-model',
      ls_temperature: 0.5,
    })
  })

  it('preserves thinking through a real LangChain tool loop', async () => {
    const { model, create } = qwenModel()
    create.mockResolvedValueOnce(
      chunks(chunk({ content: '<think>Need the lookup.</think>' + toolOutput }, 'stop')),
    )
    create.mockResolvedValueOnce(
      chunks({
        ...chunk({ content: '<think>Use the result.</think>The answer is 42.' }, 'stop'),
        id: 'answer-response',
      }),
    )
    const agent = createAgent({ model, tools: [lookup] })
    const result = await agent.invoke({ messages: [new HumanMessage('Look up A7.')] })
    expect(result.messages.map((message) => message.type)).toEqual(['human', 'ai', 'tool', 'ai'])
    const call = result.messages[1] as AIMessage
    expect(call.contentBlocks).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'Need the lookup.' }),
    )
    expect(call.tool_calls?.[0]).toMatchObject({ name: 'lookup', args: { code: 'A7' } })
    expect(result.messages.at(-1)?.text).toBe('The answer is 42.')
    expect(create.mock.calls[1]?.[0]).toMatchObject({
      messages: expect.arrayContaining([
        expect.objectContaining({
          role: 'assistant',
          content: expect.stringContaining('<think>\nNeed the lookup.\n</think>'),
        }),
        { role: 'user', content: '<tool_response>\nResult A7: 42\n</tool_response>' },
      ]),
    })
  })

  it('keeps tool images in the next request for custom vision models', async () => {
    const { engine, create } = setup()
    const model = new ChatWebLLM({
      engine,
      modelId: 'custom-vision',
      profile: { toolCalling: true, imageInputs: true, imageToolMessage: true },
    })
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: 'call-image',
                type: 'function',
                function: {
                  name: 'inspect_image',
                  arguments: '{"question":"What is shown?"}',
                },
              },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    create.mockResolvedValueOnce(
      chunks({ ...chunk({ content: 'A cat.' }, 'stop'), id: 'answer-response' }),
    )
    const inspectImage = tool(() => [{ type: 'image', data: 'AA==', mimeType: 'image/png' }], {
      name: 'inspect_image',
      description: 'Return the image to inspect.',
      schema: z.object({ question: z.string() }),
    })
    const agent = createAgent({
      model,
      tools: [inspectImage],
      middleware: [createUnsupportedContentMiddleware()],
    })

    const result = await agent.invoke({ messages: [new HumanMessage('Look at this image.')] })
    const toolResult = result.messages.find(ToolMessage.isInstance)

    expect(result.messages.map((message) => message.type)).toEqual(['human', 'ai', 'tool', 'ai'])
    expect(toolResult?.content).toEqual([{ type: 'image', data: 'AA==', mimeType: 'image/png' }])
    expect(create).toHaveBeenCalledTimes(2)
    expect(create.mock.calls[1]?.[0].messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: 'tool',
          tool_call_id: toolResult?.tool_call_id,
          content: '',
        }),
        {
          role: 'user',
          content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }],
        },
      ]),
    )
  })

  it('uses the complete default Deep Agents middleware and built-in tools', async () => {
    const { model, create } = qwenModel()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Read the file.</think><tool_call>\n<function=read_file>\n<parameter=file_path>\n/answer.txt\n</parameter>\n</function>\n</tool_call>',
          },
          'stop',
        ),
      ),
    )
    create.mockResolvedValueOnce(
      chunks({
        ...chunk({ content: '<think>The file says 42.</think>The answer is 42.' }, 'stop'),
        id: 'answer-response',
      }),
    )
    const agent = createDeepAgent({ model })
    const result = await agent.invoke({
      messages: [{ role: 'user', content: 'Read /answer.txt and tell me the answer.' }],
      files: {
        '/answer.txt': { content: ['42'], created_at: '2026-10-19', modified_at: '2026-10-19' },
      },
    })
    expect(result.messages.at(-1)?.text).toBe('The answer is 42.')
    const call = result.messages.find(
      (message) => AIMessage.isInstance(message) && message.tool_calls?.length,
    ) as AIMessage
    expect(call.tool_calls?.[0]).toMatchObject({
      name: 'read_file',
      args: { file_path: '/answer.txt' },
    })
    const first = create.mock.calls[0]?.[0]
    expect(first?.messages[0]?.content).toContain('write_file')
    expect(first?.messages[0]?.content).toContain('task')
    expect(
      create.mock.calls[1]?.[0].messages.some(
        (message) =>
          message.role === 'user' &&
          typeof message.content === 'string' &&
          message.content.includes('42'),
      ),
    ).toBe(true)
  })

  it('accepts Zod tools and structured output through the standard API', async () => {
    const { model, create } = qwenModel()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<tool_call>\n<function=extract>\n<parameter=answer>\n42\n</parameter>\n</function>\n</tool_call>',
          },
          'stop',
        ),
      ),
    )
    expect(
      await model
        .withStructuredOutput(z.object({ answer: z.number() }))
        .invoke('Return the answer.'),
    ).toEqual({ answer: 42 })
    expect(() => model.bindTools([lookup, lookup])).toThrow('unique')
  })

  it('splits thinking without putting reasoning into answer text', async () => {
    const { model } = qwenModel(chunk({ content: '<think>A thought.</think>Answer' }, 'stop'))
    let result: AIMessageChunk | undefined
    for await (const item of await model.stream('Hi')) result = result ? result.concat(item) : item
    expect(result?.text).toBe('Answer')
    expect(result?.contentBlocks).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'A thought.' }),
    )
  })

  it('does not replay thoughts from past user turns or change stored history', async () => {
    const { model, create } = qwenModel(chunk({ content: 'Okay' }, 'stop'))
    const history = [
      new HumanMessage('One'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: 'Old thought' },
          { type: 'text', text: 'One answer' },
        ],
      }),
      new HumanMessage('Two'),
    ]
    await model.invoke(history)
    expect(JSON.stringify(create.mock.calls[0]?.[0])).not.toContain('Old thought')
    expect(history[1]?.content).toContainEqual({ type: 'reasoning', reasoning: 'Old thought' })
  })

  it('preserves system messages, tool IDs, and valid normalized image content', async () => {
    const { model, create } = setup(chunk({ content: 'Okay' }, 'stop'))
    await model.invoke([
      new SystemMessage('One'),
      new SystemMessage('Two'),
      new HumanMessage({
        content: [
          { type: 'text', text: 'First' },
          { type: 'text', text: 'Last' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
        ],
      }),
      new AIMessage({
        content: '',
        tool_calls: [{ name: 'lookup', id: 'call-7', args: { code: 'A7' } }],
      }),
      new ToolMessage({ content: '42', tool_call_id: 'call-7' }),
    ])
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      messages: [
        { role: 'system', content: 'One\n\nTwo' },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'FirstLast' },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
          ],
        },
        {
          role: 'assistant',
          tool_calls: [{ id: 'call-7', function: { name: 'lookup', arguments: '{"code":"A7"}' } }],
        },
        { role: 'tool', content: '42', tool_call_id: 'call-7' },
      ],
    })
  })

  it.each(['video', 'file', 'unknown'])(
    'rejects unsupported %s content instead of losing it',
    async (type) => {
      const { model, create } = setup()
      await expect(model.invoke([new HumanMessage({ content: [{ type }] })])).rejects.toThrow(
        'cannot use',
      )
      expect(create).not.toHaveBeenCalled()
    },
  )

  it('handles empty final messages and rejects a missing stop marker', async () => {
    const { model, create } = setup(chunk({}, 'stop'))
    expect((await model.invoke('Hi')).text).toBe('')
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Unfinished' })))
    await expect(model.invoke('Hi')).rejects.toThrow('without a stop')
  })

  it('never runs a truncated native tool call', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call',
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
          ],
        },
        'length',
      ),
    )
    const result = await model.bindTools([lookup]).invoke('Look up A7.')
    expect(result.tool_calls).toEqual([])
    expect(result.response_metadata.finish_reason).toBe('length')
  })

  it('marks a truncated native call as invalid in content streams', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call',
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
          ],
        },
        'length',
      ),
    )
    const stream = model.streamEvents('Look up A7.', { tools: [lookup] })
    const [events, calls] = await Promise.all([collect(stream), stream.toolCalls])

    expect(calls).toEqual([])
    expect(events).toContainEqual(
      expect.objectContaining({
        event: 'content-block-finish',
        content: expect.objectContaining({ type: 'invalid_tool_call' }),
      }),
    )
    expect(
      events.some(
        (event) => event.event === 'content-block-finish' && event.content.type === 'tool_call',
      ),
    ).toBe(false)
  })

  it.each([
    ['Qwen2.5-3B-Instruct-q4f16_1-MLC', '<tool_call>{"name":"lookup","arguments":{"code":"A'],
    ['Qwen3.5-0.8B-q4f16_1-MLC', '<tool_call><function=lookup><parameter=code>A'],
  ])('does not expose or run a truncated %s tool call', async (modelId, raw) => {
    const { engine } = setup(chunk({ content: raw }, 'length'))
    const model = new ChatWebLLM({ engine, modelId })
    const result = await model.bindTools([lookup]).invoke('Look up code A7.')

    expect(result.text).toBe('')
    expect(result.tool_calls).toEqual([])
    expect(result.response_metadata.finish_reason).toBe('length')
  })

  it('does not expose earlier Qwen calls if a later call is truncated', async () => {
    const complete =
      '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
    const partial = '<tool_call><function=lookup><parameter=code>A8'
    const { engine } = setup(chunk({ content: complete + partial }, 'length'))
    const model = new ChatWebLLM({ engine, modelId: qwen })
    const result = await model.bindTools([lookup]).invoke('Look up codes A7 and A8.')

    expect(result.text).toBe('')
    expect(result.tool_calls).toEqual([])
    expect(result.response_metadata.finish_reason).toBe('length')
  })

  it.each(['lc_prefer_streaming', 'lc_prefer_chat_model_stream_events'] as const)(
    'does not run a truncated Qwen tool call with %s',
    async (preference) => {
      const complete =
        '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
      const partial = '<tool_call><function=lookup><parameter=code>A8'
      const { engine, create } = setup()
      create.mockResolvedValueOnce(
        chunks(chunk({ content: 'Checking.' + complete + partial }), chunk({}, 'length')),
      )
      create.mockResolvedValueOnce(chunks(chunk({ content: 'Done.' }, 'stop')))
      const model = new ChatWebLLM({ engine, modelId: qwen })
      const execute = vi.fn(() => 'Done')
      const lookup = tool(execute, { name: 'lookup', schema: z.object({ code: z.string() }) })
      const callbacks = CallbackManager.fromHandlers({
        handleLLMNewToken: () => {},
        handleChatModelStreamEvent: () => {},
      })
      const [handler] = callbacks.handlers
      if (!handler) throw new Error('Missing callback handler.')
      Object.assign(handler, { [preference]: true })

      const result = await createAgent({ model, tools: [lookup] }).invoke(
        { messages: [new HumanMessage('Look up codes A7 and A8.')] },
        { callbacks },
      )

      expect(execute).not.toHaveBeenCalled()
      expect(result.messages.at(-1)?.text).toBe('Checking.')
      expect(result.messages.at(-1)?.response_metadata.finish_reason).toBe('length')
      expect((await model.invoke([...result.messages, new HumanMessage('Try again.')])).text).toBe(
        'Done.',
      )
      expect(create.mock.calls[1]?.[0].messages).toContainEqual({
        role: 'assistant',
        content: 'Checking.',
      })
    },
  )

  it.each([
    ['Qwen2.5-3B-Instruct-q4f16_1-MLC', '<tool_call>{"name":"lookup","arguments":{"code":"A'],
    ['Qwen3.5-0.8B-q4f16_1-MLC', '<tool_call><function=lookup><parameter=code>A'],
  ])(
    'does not expose partial %s tool syntax before reporting cancellation',
    async (modelId, raw) => {
      const { engine } = setup(chunk({ content: raw }, 'abort'))
      const model = new ChatWebLLM({ engine, modelId })
      const output: string[] = []

      await expect(async () => {
        const stream = await model.bindTools([lookup]).stream('Look up code A7.')
        for await (const message of stream) output.push(message.text)
      }).rejects.toThrow('cancelled generation')

      expect(output.join('')).toBe('')
    },
  )

  it('keeps tool_choice none and rejects unsupported required choices', async () => {
    const { model, create, engine } = setup(chunk({ content: 'Answer' }, 'stop'))
    const tools = [lookup, tool(() => '', { name: 'other', schema: z.object({}) })]
    await model.bindTools([lookup], { tool_choice: 'none' }).invoke('Hi')
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty('tools')
    expect(create.mock.calls[0]?.[0].tool_choice).toBe('none')
    await expect(model.bindTools(tools, { tool_choice: 'required' }).invoke('Hi')).rejects.toThrow(
      'exactly one tool',
    )
    const gemma = new ChatWebLLM({ engine, modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })
    await expect(gemma.bindTools(tools, { tool_choice: 'required' }).invoke('Hi')).rejects.toThrow(
      'exactly one tool',
    )
    expect(create).toHaveBeenCalledTimes(1)
  })

  it('rejects unregistered native tool choices before generating', async () => {
    const { model, create } = setup()

    await expect(
      model.bindTools([lookup], { tool_choice: 'missing' }).invoke('Hi'),
    ).rejects.toThrow('not registered')
    await expect(model.bindTools([], { tool_choice: 'missing' }).invoke('Hi')).rejects.toThrow(
      'not registered',
    )
    expect(create).not.toHaveBeenCalled()
  })
})
