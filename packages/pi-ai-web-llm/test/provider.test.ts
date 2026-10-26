import { describe, expect, it, vi } from 'vitest'
import {
  Type,
  type Context,
  type Model,
  type Api,
  type SimpleStreamOptions,
} from '@earendil-works/pi-ai'
import { ModelType, prebuiltAppConfig, type ChatCompletionTool } from '@mlc-ai/web-llm'
import { createModels } from '@earendil-works/pi-ai/models'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { Agent } from '@earendil-works/pi-agent-core'
import { chunk, chunks, collect, setup } from './helpers.js'
import {
  createWebLLMProvider,
  type WebLLMProviderOptions,
  type WebLLMStreamOptions,
} from '../src/index.js'
import { nativeTools } from '../../../shared/web-llm/native.js'

const context: Context = { messages: [{ role: 'user', content: 'Hello', timestamp: 0 }] }
const lookup = {
  name: 'lookup',
  description: 'Look up a code.',
  parameters: Type.Object({ code: Type.String() }),
}
const tools = { ...context, tools: [lookup] }
const call = (args = '{"code":"A7"}', name = 'lookup', id = 'call-1') =>
  chunk(
    { tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: args } }] },
    'tool_calls',
  )

describe('Pi AI WebLLM provider', () => {
  it('forwards WebLLM usage stream options', async () => {
    const { models, model, create } = setup([chunk({ content: 'Answer' }, 'stop')])

    await models.complete(model, context, {
      request: { stream_options: { include_usage: false } },
    })

    expect(create.mock.calls[0]?.[0].stream_options).toEqual({ include_usage: false })
  })

  it.each([false, true])(
    'uses Pi request telemetry without changing model I/O (simple=%s)',
    async (simple) => {
      const native = {
        ...chunk({ content: 'Private answer' }, 'stop'),
        usage: {
          prompt_tokens: 4,
          completion_tokens: 1,
          total_tokens: 5,
          extra: {
            e2e_latency_s: 0,
            prefill_tokens_per_s: 0,
            decode_tokens_per_s: 0,
            time_to_first_token_s: 0,
            time_per_output_token_s: 0,
          },
        },
      }
      const { models, model, create } = setup([native])
      let settled!: () => void
      const finished = new Promise<void>((resolve) => {
        settled = resolve
      })
      const telemetry: NonNullable<WebLLMStreamOptions['telemetry']> = {
        startSpan: vi.fn(async (_options, run) => {
          try {
            return await run(telemetry)
          } finally {
            settled()
          }
        }),
        addLink: vi.fn(),
        addEvent: vi.fn(),
        setAttributes: vi.fn(),
        setStatus: vi.fn(),
      }
      const complete = simple ? models.completeSimple.bind(models) : models.complete.bind(models)
      const baseline = await complete(model, context)
      const result = await complete(model, context, {
        telemetry,
        identity: { userAgent: 'short', attribution: null },
      })

      await finished
      expect(create.mock.calls[1]).toEqual(create.mock.calls[0])
      expect(result.content).toEqual(baseline.content)
      expect(result.usage).toEqual(baseline.usage)
      expect(telemetry.startSpan).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'pi.ai.request',
          attributes: { provider: 'webllm', model: 'loaded-model', api: 'webllm' },
        }),
        expect.any(Function),
      )
      expect(telemetry.setAttributes).toHaveBeenCalledWith(
        expect.objectContaining({ stop: 'stop', 'usage.input': 4, 'usage.output': 1 }),
      )
      expect(
        JSON.stringify([
          vi.mocked(telemetry.addEvent).mock.calls,
          vi.mocked(telemetry.setAttributes).mock.calls,
        ]),
      ).not.toContain('Private answer')
    },
  )

  it.each([false, true])('preserves native response metadata (simple=%s)', async (simple) => {
    const native = [
      { ...chunk({ content: 'Hello' }), id: 'native-response', model: 'served-model' },
      { ...chunk({}, 'length'), id: 'native-response', model: 'served-model' },
      { ...chunk(), id: 'native-response', model: 'served-model', choices: [] },
    ]
    const onChunk = vi.fn()
    const { models, model } = setup(native, { onChunk })
    const result = simple
      ? await models.completeSimple(model, context)
      : await models.complete(model, context)

    expect(result).toMatchObject({
      model: 'loaded-model',
      responseModel: 'served-model',
      responseId: 'native-response',
      rawStopReason: 'length',
      stopReason: 'length',
      content: [{ type: 'text', text: 'Hello' }],
    })
    expect(onChunk.mock.calls.map(([chunk]) => chunk)).toEqual(native)
  })

  it.each([false, true])(
    'keeps native response identity on stream errors (simple=%s)',
    async (simple) => {
      const { models, model, create } = setup()
      create.mockResolvedValueOnce(
        (async function* () {
          yield { ...chunk({ content: 'Received' }), id: 'partial-response', model: 'served-model' }
          throw new Error('Native stream failed')
        })(),
      )
      const result = simple
        ? await models.completeSimple(model, context)
        : await models.complete(model, context)

      expect(result).toMatchObject({
        responseId: 'partial-response',
        responseModel: 'served-model',
        stopReason: 'error',
        errorMessage: 'Native stream failed',
        content: [{ type: 'text', text: 'Received' }],
      })
      expect(result).not.toHaveProperty('rawStopReason')
    },
  )

  it('registers a keyless model and streams text with native usage and zero cost', async () => {
    const final = {
      ...chunk({}, 'stop'),
      usage: {
        prompt_tokens: 12,
        completion_tokens: 3,
        total_tokens: 15,
        extra: {
          e2e_latency_s: 0,
          prefill_tokens_per_s: 0,
          decode_tokens_per_s: 0,
          time_to_first_token_s: 0,
          time_per_output_token_s: 0,
        },
      },
    }
    const { models, model, create } = setup([
      chunk({ content: 'Hel' }),
      chunk({ content: 'lo' }),
      final,
    ])
    expect(models.getModel('webllm', model.id)).toEqual(model)
    const stream = models.stream(model, context, {
      temperature: 0.2,
      maxTokens: 50,
      request: { seed: 7 },
    })
    const events = await collect(stream)
    expect(events.map((event) => event.type)).toEqual([
      'start',
      'text_start',
      'text_delta',
      'text_delta',
      'text_end',
      'done',
    ])
    const result = await stream.result()
    expect(result).toMatchObject({
      api: 'webllm',
      provider: 'webllm',
      model: model.id,
      stopReason: 'stop',
      content: [{ type: 'text', text: 'Hello' }],
      usage: {
        input: 12,
        output: 3,
        totalTokens: 15,
        cacheRead: 0,
        cacheWrite: 0,
        cost: { total: 0 },
      },
    })
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: model.id,
      n: 1,
      stream: true,
      temperature: 0.2,
      max_tokens: 50,
      seed: 7,
    })
  })

  it.each(['engine', 'modelId', 'onChunk'] as const)(
    'keeps its configured %s when the caller reuses its options object',
    async (field) => {
      const original = setup()
      const other = setup([chunk({ content: 'Other' }, 'stop')])
      const onChunk = vi.fn()
      const replacement = vi.fn()
      const options: WebLLMProviderOptions = {
        engine: original.engine,
        modelId: 'loaded-model',
        contextWindow: 8192,
        onChunk,
      }
      const provider = createWebLLMProvider(options)
      const models = createModels()
      models.setProvider(provider)
      const model = provider.getModels()[0]!
      const before = await models.completeSimple(model, context)
      if (field === 'engine') options.engine = other.engine
      if (field === 'modelId') options.modelId = 'other-model'
      if (field === 'onChunk') options.onChunk = replacement
      const after = await models.completeSimple(model, context)

      expect(after).toMatchObject({ model: model.id, stopReason: 'stop', content: before.content })
      expect(original.create).toHaveBeenCalledTimes(2)
      expect(original.create.mock.calls[1]?.[0]).toEqual(original.create.mock.calls[0]?.[0])
      expect(other.create).not.toHaveBeenCalled()
      expect(onChunk).toHaveBeenCalledTimes(2)
      expect(replacement).not.toHaveBeenCalled()
    },
  )

  it.each(['id', 'api'] as const)(
    'rejects a changed registered model %s before using the engine',
    async (field) => {
      const { provider, model, create } = setup()
      const original = { id: model.id, api: model.api }
      const changed: Model<Api> = model
      if (field === 'id') changed.id = 'other-model'
      else changed.api = 'openai-responses'
      const result = await provider.streamSimple(model, normalizeContext(context)).result()

      expect(result).toMatchObject({
        stopReason: 'error',
        errorMessage: 'This provider only serves its configured WebLLM model.',
      })
      expect(create).not.toHaveBeenCalled()
      Object.assign(model, original)
      expect(
        (await provider.streamSimple(model, normalizeContext(context)).result()).content,
      ).toEqual([{ type: 'text', text: 'Hello' }])
    },
  )

  it('completes even when its events are not consumed', async () => {
    const { models, model } = setup()
    expect((await models.streamSimple(model, context).result()).stopReason).toBe('stop')
  })

  it.each([
    'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  ])('%s keeps its system prompt', async (modelId) => {
    const { models, model, create } = setup(undefined, { modelId })

    await models.completeSimple(model, context)

    expect(create.mock.calls[0]?.[0].messages[0]).toEqual({
      role: 'system',
      content: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
    })
  })

  it('keeps the Qwen tool protocol when Pi supplies an empty system message', async () => {
    const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
    const { models, model, create } = setup(undefined, { modelId })

    await models.completeSimple(model, tools)

    const system = create.mock.calls[0]?.[0].messages[0]?.content as string
    expect(system).toContain('You may call one or more functions')
    expect(system).not.toContain('You are Qwen')
  })

  it.each([
    'loaded-model',
    'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    'Qwen3-8B-q4f16_1-MLC',
    'Qwen3.5-4B-q4f16_1-MLC',
    'gemma-4-E2B-it-q4f16_1-MLC',
  ])(
    '%s sends identical native requests for explicit and default auto tool choice',
    async (modelId) => {
      const { models, model, engine, create } = setup(undefined, { modelId })
      const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
        (async function* () {
          yield {
            id: 'raw',
            object: 'text_completion' as const,
            created: 0,
            model: modelId,
            choices: [{ index: 0, text: 'Hello', finish_reason: 'stop' as const, logprobs: null }],
          }
        })(),
      )
      expect((await models.completeSimple(model, tools)).stopReason).toBe('stop')
      expect((await models.completeSimple(model, tools, { toolChoice: 'auto' })).stopReason).toBe(
        'stop',
      )
      const calls = modelId.startsWith('gemma-') ? raw.mock.calls : create.mock.calls
      expect(calls).toHaveLength(2)
      expect(calls[1]?.[0]).toEqual(calls[0]?.[0])
    },
  )

  it('replays current system sections and tool declarations without mutating the transcript', async () => {
    const { models, model, create } = setup()
    const input: Context = {
      systemPrompt: 'Base',
      tools: [lookup],
      messages: [
        ...context.messages,
        {
          role: 'system',
          content: 'Extra',
          sections: { files: 'Current files' },
          toolsRemoved: [{ name: 'lookup' }],
          toolsAdded: [{ ...lookup, name: 'search' }],
          timestamp: 1,
        },
      ],
    }
    const before = structuredClone(input)
    await models.completeSimple(model, input)
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      messages: [
        { role: 'system', content: 'Base\n\nExtra\n\nCurrent files' },
        { role: 'user', content: 'Hello' },
      ],
      tools: [
        {
          function: {
            name: 'search',
            parameters: { type: 'object', properties: { code: { type: 'string' } } },
          },
        },
      ],
    })
    expect(input).toEqual(before)
  })

  it('streams native tool deltas into a complete Pi tool call', async () => {
    const { models, model } = setup([
      chunk({
        tool_calls: [{ index: 0, id: 'call-1', function: { name: 'look', arguments: '{"code":' } }],
      }),
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'up', arguments: '"A7"}' } }] },
        'tool_calls',
      ),
    ])
    const stream = models.streamSimple(model, tools)
    expect((await collect(stream)).map((event) => event.type)).toEqual([
      'start',
      'toolcall_start',
      'toolcall_delta',
      'toolcall_delta',
      'toolcall_end',
      'done',
    ])
    expect((await stream.result()).content).toEqual([
      { type: 'toolCall', id: 'call-1', name: 'lookup', arguments: { code: 'A7' } },
    ])
  })

  it.each([true, false])(
    'runs the Pi Agent tool loop when the native ID is present=%s',
    async (includeNativeId) => {
      const { models, model, create } = setup()
      const toolCall = chunk(
        {
          tool_calls: [
            {
              index: 0,
              ...(includeNativeId && { id: 'call-1' }),
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
          ],
        },
        'tool_calls',
      )
      create
        .mockResolvedValueOnce(chunks(toolCall))
        .mockResolvedValueOnce(chunks(chunk({ content: 'ORCHID-42' }, 'stop')))
      const executedIds: string[] = []
      const execute = vi.fn(async (id: string, args: unknown) => {
        executedIds.push(id)
        expect(args).toEqual({ code: 'A7' })
        return { content: [{ type: 'text' as const, text: 'ORCHID-42' }], details: {} }
      })
      const agent = new Agent({
        initialState: { model, tools: [{ ...lookup, label: 'Lookup', execute }] },
        streamFn: models.streamSimple.bind(models),
      })
      await agent.prompt('Look up A7.')

      expect(execute).toHaveBeenCalledOnce()
      const executedId = executedIds[0]!
      if (includeNativeId) expect(executedId).toBe('call-1')
      else
        expect(executedId).toMatch(
          /^[\da-f]{8}-[\da-f]{4}-4[\da-f]{3}-[89ab][\da-f]{3}-[\da-f]{12}$/i,
        )
      expect(create.mock.calls[1]?.[0]).toMatchObject({
        messages: expect.arrayContaining([
          { role: 'tool', tool_call_id: executedId, content: 'ORCHID-42' },
        ]),
      })
      expect(agent.state.messages.at(-1)).toMatchObject({
        role: 'assistant',
        stopReason: 'stop',
        content: [{ type: 'text', text: 'ORCHID-42' }],
      })
    },
  )

  it('does not execute a tool from a failed streamed call', async () => {
    const { models, model, create } = setup([
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{' },
            },
          ],
        },
        'tool_calls',
      ),
    ])
    const execute = vi.fn(async () => ({ content: [], details: {} }))
    const agent = new Agent({
      initialState: { model, tools: [{ ...lookup, label: 'Lookup', execute }] },
      streamFn: models.streamSimple.bind(models),
    })

    await agent.prompt('Look up A7.')

    expect(execute).not.toHaveBeenCalled()
    expect(create).toHaveBeenCalledOnce()
    expect(agent.state.messages.at(-1)).toMatchObject({
      role: 'assistant',
      stopReason: 'error',
      content: [{ type: 'toolCall', id: 'call-1', name: 'lookup' }],
    })
  })

  it('rejects a tool call when its native ID changes between chunks', async () => {
    const { models, model } = setup([
      chunk({
        tool_calls: [
          {
            index: 0,
            id: 'call-1',
            type: 'function',
            function: { name: 'look', arguments: '{"code":' },
          },
        ],
      }),
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call-2',
              function: { name: 'up', arguments: '"A7"}' },
            },
          ],
        },
        'tool_calls',
      ),
    ])

    expect(await models.completeSimple(model, tools)).toMatchObject({
      stopReason: 'error',
      errorMessage: expect.stringContaining('changed a tool call ID'),
    })
  })

  it('rejects parallel tool calls with the same native ID', async () => {
    const { models, model } = setup([
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
            {
              index: 1,
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"B8"}' },
            },
          ],
        },
        'tool_calls',
      ),
    ])

    expect(await models.completeSimple(model, tools)).toMatchObject({
      stopReason: 'error',
      errorMessage: 'WebLLM returned duplicate tool call IDs.',
      content: [],
    })
  })

  it('rejects a tool-use stop without any tool calls', async () => {
    const { models, model } = setup([chunk({}, 'tool_calls')])

    expect(await models.completeSimple(model, tools)).toMatchObject({
      stopReason: 'error',
      errorMessage: 'WebLLM returned no tool calls.',
      content: [],
    })
  })

  it.each(['Qwen3-8B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s separates thinking, tools and final text through the shared protocol',
    async (modelId) => {
      const xml = modelId.startsWith('Qwen3.5-')
      const output =
        '<think>Check</think>' +
        (xml
          ? '<tool_call>\n<function=lookup>\n<parameter=code>\nA7\n</parameter>\n</function>\n</tool_call>'
          : '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>')
      const { models, model, create } = setup(
        [chunk({ content: output.slice(0, 10) }), chunk({ content: output.slice(10) }, 'stop')],
        { modelId },
      )
      const result = await models.completeSimple(model, tools, { reasoning: 'low' })
      expect(result.stopReason).toBe('toolUse')
      expect(result.content).toEqual([
        { type: 'thinking', thinking: 'Check' },
        expect.objectContaining({ type: 'toolCall', name: 'lookup', arguments: { code: 'A7' } }),
      ])
      expect(create.mock.calls[0]?.[0]).toHaveProperty('extra_body.enable_thinking', true)
    },
  )

  it('constrains verified Qwen tool arguments on request and allows disabling it per call', async () => {
    const output = '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
    const { models, model, create } = setup([chunk({ content: output }, 'stop')], {
      modelId: 'Qwen3-4B-q4f16_1-MLC',
    })

    const result = await models.complete(model, tools, { request: { constrainedToolCalls: true } })
    await models.complete(model, tools, { request: { constrainedToolCalls: false } })

    expect(result.stopReason).toBe('toolUse')
    expect(result.content).toContainEqual(
      expect.objectContaining({ type: 'toolCall', name: 'lookup', arguments: { code: 'A7' } }),
    )
    expect(create.mock.calls[0]?.[0].response_format?.type).toBe('structural_tag')
    expect(create.mock.calls[1]?.[0]).not.toHaveProperty('response_format')
    expect(create.mock.calls[1]?.[0]).not.toHaveProperty('constrainedToolCalls')
  })

  it('keeps constrained tool parsing when onPayload only reorders JSON keys', async () => {
    const output = '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
    const { models, model, create } = setup([chunk({ content: output }, 'stop')], {
      modelId: 'Qwen3-4B-q4f16_1-MLC',
    })
    const result = await models.complete(model, tools, {
      request: { constrainedToolCalls: true },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        const format = 'response_format' in payload ? payload.response_format : undefined
        const reordered = reverseKeys(format)
        expect(JSON.stringify(reordered)).not.toBe(JSON.stringify(format))
        return {
          ...payload,
          response_format: reordered,
        }
      },
    })

    expect(result.stopReason).toBe('toolUse')
    expect(result.content).toContainEqual(
      expect.objectContaining({ type: 'toolCall', name: 'lookup', arguments: { code: 'A7' } }),
    )
    expect(create.mock.calls[0]?.[0].response_format).toEqual(
      expect.objectContaining({ type: 'structural_tag' }),
    )
  })

  it('keeps constrained tool parsing when onPayload adds an undefined optional field', async () => {
    const output = '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
    const { models, model, create } = setup([chunk({ content: output }, 'stop')], {
      modelId: 'Qwen3-4B-q4f16_1-MLC',
    })
    let originalFormat: string | undefined
    const result = await models.complete(model, tools, {
      request: { constrainedToolCalls: true },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        const format = 'response_format' in payload ? payload.response_format : undefined
        if (!format) throw new Error('Expected constrained tool format.')
        originalFormat = JSON.stringify(format)
        return { ...payload, response_format: { ...format, schema: undefined } as typeof format }
      },
    })

    expect(result.stopReason).toBe('toolUse')
    expect(result.content).toContainEqual(
      expect.objectContaining({ type: 'toolCall', name: 'lookup', arguments: { code: 'A7' } }),
    )
    expect(create.mock.calls).toHaveLength(1)
    expect(JSON.stringify(create.mock.calls[0]?.[0].response_format)).toBe(originalFormat)
  })

  it.each(['{', '[]', '{"code":1e999}', 'null'])(
    'does not publish executable tool calls for invalid arguments %s',
    async (args) => {
      const { models, model } = setup([call(args)])
      const stream = models.streamSimple(model, tools)
      expect((await collect(stream)).some((event) => event.type === 'toolcall_end')).toBe(false)
      expect((await stream.result()).stopReason).toBe('error')
    },
  )

  it('does not complete a valid Qwen call before a later unknown call fails', async () => {
    const valid =
      '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
    const invalid = valid.replace('<function=lookup>', '<function=missing>')
    const { models, model } = setup(
      [chunk({ content: valid }), chunk({ content: invalid }, 'stop')],
      { modelId: 'Qwen3.5-4B-q4f16_1-MLC' },
    )
    const stream = models.streamSimple(model, tools)
    const events = await collect(stream)

    expect(events.some((event) => event.type === 'toolcall_start')).toBe(true)
    expect(events.filter((event) => event.type === 'toolcall_end')).toEqual([])
    expect((await stream.result()).stopReason).toBe('error')
  })

  it('does not execute a tool the model was not offered', async () => {
    const { models, model } = setup([call('{}', 'unregistered')])
    expect(await models.completeSimple(model, tools)).toMatchObject({
      stopReason: 'error',
      errorMessage: expect.stringContaining('unknown tool'),
    })
  })

  it('enforces toolChoice none and leaves truncated calls unfinished', async () => {
    const { models, model, create } = setup([call()])
    expect((await models.completeSimple(model, tools, { toolChoice: 'none' })).stopReason).toBe(
      'error',
    )
    expect(create.mock.calls[0]?.[0]).toHaveProperty('tool_choice', 'none')
    const incomplete = call('{')
    create.mockResolvedValueOnce(
      chunks({
        ...incomplete,
        choices: [{ ...incomplete.choices[0]!, finish_reason: 'length' }],
      }),
    )
    const truncated = await models.completeSimple(model, tools)
    expect(truncated).toMatchObject({
      stopReason: 'length',
      content: [{ type: 'toolCall', id: 'call-1', name: 'lookup', arguments: {} }],
    })
  })

  it('selects a registered tool with WebLLM native tool choice', async () => {
    const { models, model, create } = setup([call()])
    const result = await models.complete(model, tools, {
      toolChoice: { type: 'function', function: { name: 'lookup' } },
    })

    expect(create.mock.calls[0]?.[0].tool_choice).toEqual({
      type: 'function',
      function: { name: 'lookup' },
    })
    expect(result).toMatchObject({
      stopReason: 'toolUse',
      content: [expect.objectContaining({ type: 'toolCall', name: 'lookup' })],
    })
  })

  it('uses the native tool choice selected by onPayload', async () => {
    const { models, model, create } = setup([chunk({ content: 'No tool.' }, 'stop')])
    const result = await models.complete(model, tools, {
      toolChoice: { type: 'function', function: { name: 'lookup' } },
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, tool_choice: 'none' }
      },
    })

    expect(result.errorMessage).toBeUndefined()
    expect(result.stopReason).toBe('stop')
    expect(create.mock.calls[0]?.[0].tool_choice).toBe('none')
  })

  it('suppresses native tool JSON when onPayload adds a registered tool', async () => {
    const callChunk = call()
    const output = '[{"name":"lookup","arguments":{"code":"A7"}}]'
    const { models, model } = setup([
      chunk({ ...callChunk.choices[0]!.delta, content: output }, 'tool_calls'),
    ])
    const result = await models.complete(model, tools, {
      toolChoice: 'none',
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return {
          ...payload,
          tools: nativeTools([
            {
              name: 'lookup',
              description: 'Look up a code.',
              inputSchema: {
                type: 'object',
                properties: { code: { type: 'string' } },
                required: ['code'],
                additionalProperties: false,
              },
            },
          ]),
          tool_choice: 'auto',
        }
      },
    })

    expect(result.stopReason).toBe('toolUse')
    expect(result.content).toEqual([
      expect.objectContaining({ type: 'toolCall', name: 'lookup', arguments: { code: 'A7' } }),
    ])
  })

  it('accepts the tool selected by onPayload', async () => {
    const { models, model, create } = setup([call()])
    const result = await models.complete(model, tools, {
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return {
          ...payload,
          tool_choice: { type: 'function', function: { name: 'lookup' } },
        }
      },
    })

    expect(create.mock.calls[0]?.[0].tool_choice).toEqual({
      type: 'function',
      function: { name: 'lookup' },
    })
    expect(result).toMatchObject({
      stopReason: 'toolUse',
      content: [expect.objectContaining({ type: 'toolCall', name: 'lookup' })],
    })
  })

  it('rejects a tool outside the native choice selected by onPayload', async () => {
    const other = { ...lookup, name: 'search' }
    const { models, model, create } = setup([call('{}', 'search')])
    const result = await models.complete(
      model,
      { ...tools, tools: [lookup, other] },
      {
        onPayload: (payload) => {
          if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
          return {
            ...payload,
            tool_choice: { type: 'function', function: { name: 'lookup' } },
          }
        },
      },
    )

    expect(result).toMatchObject({ stopReason: 'error' })
    expect(create.mock.calls[0]?.[0].tool_choice).toEqual({
      type: 'function',
      function: { name: 'lookup' },
    })
  })

  it('rejects a tool choice outside the prepared tool list', async () => {
    const { models, model, create } = setup()
    const result = await models.complete(model, tools, {
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return {
          ...payload,
          tool_choice: { type: 'function', function: { name: 'missing' } },
        }
      },
    })

    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: 'onPayload selected a tool that is not in its tools list.',
    })
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects tools that are not registered with Pi before inference', async () => {
    const { models, model, create } = setup()
    const result = await models.complete(model, tools, {
      onPayload: (payload) => {
        if (
          !payload ||
          typeof payload !== 'object' ||
          !('tools' in payload) ||
          !Array.isArray(payload.tools)
        )
          throw new Error('Expected registered tools in the native request.')
        const registeredTools = payload.tools as ChatCompletionTool[]
        return {
          ...payload,
          tools: [
            ...registeredTools,
            {
              type: 'function',
              function: { name: 'missing', description: 'Unregistered.', parameters: {} },
            },
          ],
        }
      },
    })

    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: 'onPayload cannot add tools that Pi has not registered.',
    })
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects duplicate registered tools in the prepared request', async () => {
    const { models, model, create } = setup()
    const result = await models.complete(model, tools, {
      onPayload: (payload) => {
        if (
          !payload ||
          typeof payload !== 'object' ||
          !('tools' in payload) ||
          !Array.isArray(payload.tools)
        )
          throw new Error('Expected registered tools in the native request.')
        const registeredTools = payload.tools as ChatCompletionTool[]
        return { ...payload, tools: [...registeredTools, registeredTools[0]!] }
      },
    })

    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: 'onPayload must keep tool names unique.',
    })
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects a tool removed by onPayload', async () => {
    const other = { ...lookup, name: 'search' }
    const { models, model, create } = setup([call('{}', 'search')])
    const result = await models.complete(
      model,
      { ...tools, tools: [lookup, other] },
      {
        onPayload: (payload) => {
          if (
            !payload ||
            typeof payload !== 'object' ||
            !('tools' in payload) ||
            !Array.isArray(payload.tools)
          )
            throw new Error('Expected registered tools in the native request.')
          const registeredTools = payload.tools as ChatCompletionTool[]
          return {
            ...payload,
            tools: registeredTools.filter((tool) => tool.function.name !== 'search'),
          }
        },
      },
    )

    expect(result).toMatchObject({ stopReason: 'error' })
    expect(create.mock.calls[0]?.[0].tools?.map((tool) => tool.function.name)).toEqual(['lookup'])
  })

  it('accepts a remaining tool when onPayload narrows the registered list', async () => {
    const other = { ...lookup, name: 'search' }
    const { models, model, create } = setup([call('{}', 'lookup')])
    const result = await models.complete(
      model,
      { ...tools, tools: [lookup, other] },
      {
        onPayload: (payload) => {
          if (
            !payload ||
            typeof payload !== 'object' ||
            !('tools' in payload) ||
            !Array.isArray(payload.tools)
          )
            throw new Error('Expected registered tools in the native request.')
          const registeredTools = payload.tools as ChatCompletionTool[]
          return {
            ...payload,
            tools: registeredTools.filter((tool) => tool.function.name !== 'search'),
          }
        },
      },
    )

    expect(create.mock.calls[0]?.[0].tools?.map((tool) => tool.function.name)).toEqual(['lookup'])
    expect(result).toMatchObject({
      stopReason: 'toolUse',
      content: [expect.objectContaining({ type: 'toolCall', name: 'lookup' })],
    })
  })

  it('uses the named choice in the Qwen protocol and validates its tool call', async () => {
    const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
    const output =
      '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
    const { models, model, create } = setup([chunk({ content: output }, 'stop')], { modelId })
    const result = await models.complete(model, tools, {
      toolChoice: { type: 'function', function: { name: 'lookup' } },
    })

    expect(create.mock.calls[0]?.[0].messages[0]).toMatchObject({
      role: 'system',
      content: expect.stringContaining('Call only the lookup function.'),
    })
    expect(result).toMatchObject({
      stopReason: 'toolUse',
      content: [expect.objectContaining({ type: 'toolCall', name: 'lookup' })],
    })
  })

  it('uses the named choice in the Gemma protocol', async () => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    const { models, model, engine } = setup(undefined, { modelId })
    const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield {
          id: 'raw',
          object: 'text_completion' as const,
          model: modelId,
          created: 0,
          choices: [{ index: 0, text: '"code":"A7"}<tool_call|>', finish_reason: 'stop' as const }],
        }
      })(),
    )
    const result = await models.complete(model, tools, {
      toolChoice: { type: 'function', function: { name: 'lookup' } },
    })

    expect(raw.mock.calls[0]?.[0].prompt).toContain('<|tool_call>call:lookup{')
    expect(result).toMatchObject({
      stopReason: 'toolUse',
      content: [expect.objectContaining({ type: 'toolCall', name: 'lookup' })],
    })
  })

  it('rejects an unregistered named tool before inference', async () => {
    const { models, model, create } = setup([call()])
    const result = await models.complete(model, tools, {
      toolChoice: { type: 'function', function: { name: 'missing' } },
    })

    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: expect.stringContaining('not registered'),
    })
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects a different registered tool returned for a named choice', async () => {
    const other = { ...lookup, name: 'search' }
    const { models, model, create } = setup([call('{}', 'search')])
    const result = await models.complete(
      model,
      { ...tools, tools: [lookup, other] },
      {
        toolChoice: { type: 'function', function: { name: 'lookup' } },
      },
    )

    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: expect.stringContaining('outside tool_choice'),
    })
    expect(create).toHaveBeenCalledOnce()
  })

  it.each(
    [false, true].flatMap((simple) =>
      ['required', 'lookup', 'none ', null, { type: 'function', function: { name: 7 } }].map(
        (toolChoice) => ({ simple, toolChoice }),
      ),
    ),
  )(
    'rejects unsupported JavaScript toolChoice $toolChoice before inference (simple=$simple)',
    async ({ simple, toolChoice }) => {
      const onChunk = vi.fn()
      const { models, model, create } = setup([call()], {
        modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
        onChunk,
      })
      const onPayload = vi.fn()
      const onProviderStreamEvent = vi.fn()
      const options: SimpleStreamOptions = { onPayload, onProviderStreamEvent }
      Object.assign(options, { toolChoice })
      const result = simple
        ? await models.completeSimple(model, tools, options)
        : await models.complete(model, tools, options)

      expect(result).toMatchObject({
        stopReason: 'error',
        errorMessage: 'toolChoice must be auto, none, or a function choice.',
        content: [],
      })
      expect(create).not.toHaveBeenCalled()
      expect(onChunk).not.toHaveBeenCalled()
      expect(onPayload).not.toHaveBeenCalled()
      expect(onProviderStreamEvent).not.toHaveBeenCalled()
      expect((await models.completeSimple(model, tools, { toolChoice: 'auto' })).stopReason).toBe(
        'toolUse',
      )
      expect(create).toHaveBeenCalledOnce()
    },
  )

  it('rejects missing terminal chunks and preserves received text', async () => {
    const { models, model } = setup([chunk({ content: 'Partial' })])
    expect(await models.completeSimple(model, context)).toMatchObject({
      stopReason: 'error',
      content: [{ type: 'text', text: 'Partial' }],
      errorMessage: expect.stringContaining('stop reason'),
    })
  })

  it('reports native failures as Pi errors and releases the engine', async () => {
    const { models, model, create } = setup()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'Partial' })
        throw new Error('GPU lost')
      })(),
    )
    expect(await models.completeSimple(model, context)).toMatchObject({
      stopReason: 'error',
      errorMessage: 'GPU lost',
      content: [{ type: 'text', text: 'Partial' }],
    })
    expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
  })

  it('reports errors when a provider proxy rejects prototype inspection', async () => {
    const { models, model, create } = setup()
    const error = new Proxy(new TypeError('GPU lost'), {
      getPrototypeOf() {
        throw new Error('Cannot inspect provider prototype')
      },
    })
    create.mockRejectedValueOnce(error)

    const result = await models.completeSimple(model, context)
    expect(result).toMatchObject({ stopReason: 'error', errorMessage: 'GPU lost' })
  })

  it('awaits native observers and interrupts after an observer cancels', async () => {
    const { models, model, interrupt } = setup()
    const controller = new AbortController()
    const observe = vi.fn(async () => {
      await Promise.resolve()
      controller.abort()
    })
    expect(
      (
        await models.completeSimple(model, context, {
          signal: controller.signal,
          onProviderStreamEvent: observe,
        })
      ).stopReason,
    ).toBe('aborted')
    expect(observe).toHaveBeenCalledTimes(1)
    expect(interrupt).toHaveBeenCalled()
    expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
  })

  it('does not start an engine call when already aborted', async () => {
    const { provider, model, create } = setup()
    expect(
      (
        await provider
          .streamSimple(model, normalizeContext(context), { signal: AbortSignal.abort() })
          .result()
      ).stopReason,
    ).toBe('aborted')
    expect(create).not.toHaveBeenCalled()
  })

  it('serializes calls on the same engine and can cancel a queued call', async () => {
    const { models, model, create, interrupt, provider } = setup()
    let release!: () => void
    let entered!: () => void
    const started = new Promise<void>((resolve) => {
      entered = resolve
    })
    const wait = new Promise<void>((resolve) => {
      release = resolve
    })
    create.mockResolvedValueOnce(
      (async function* () {
        entered()
        await wait
        yield chunk({ content: 'First' }, 'stop')
      })(),
    )
    const first = models.completeSimple(model, context)
    await started
    const controller = new AbortController()
    const second = provider
      .streamSimple(model, normalizeContext(context), { signal: controller.signal })
      .result()
    controller.abort()
    expect((await second).stopReason).toBe('aborted')
    expect(create).toHaveBeenCalledTimes(1)
    expect(interrupt).not.toHaveBeenCalled()
    release()
    expect((await first).stopReason).toBe('stop')
    expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
  })

  it.each([0, -1, NaN, 1.5])('rejects invalid contextWindow %s', (contextWindow) => {
    const { engine } = setup()
    expect(() => createWebLLMProvider({ engine, modelId: 'model', contextWindow })).toThrow(
      'positive integer',
    )
  })

  it('rejects unsupported constrained tool schemas before native generation', async () => {
    const { models, model, create } = setup()
    const result = await models.completeSimple(model, {
      ...context,
      tools: [{ ...lookup, constrainedSampling: { type: 'json_schema', strict: 'require' } }],
    })
    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: expect.stringContaining('constrained'),
    })
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects image input on a text model instead of dropping it', async () => {
    const { models, model, create } = setup()
    const result = await models.completeSimple(model, {
      messages: [
        {
          role: 'user',
          content: [{ type: 'image', mimeType: 'image/png', data: 'AQ==' }],
          timestamp: 0,
        },
      ],
    })
    expect(result.stopReason).toBe('error')
    expect(create).not.toHaveBeenCalled()
  })

  it.each([
    ['Phi-3.5-vision-instruct-q4f16_1-MLC', {}],
    ['custom-vision', { imageInputs: true }],
  ] as const)('preserves base64 images on %s', async (modelId, options) => {
    const { models, model, create } = setup(undefined, { modelId, ...options })
    await models.completeSimple(model, {
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Color?' },
            { type: 'image', mimeType: 'image/png', data: 'AQ==' },
          ],
          timestamp: 0,
        },
      ],
    })
    expect(create.mock.calls[0]?.[0]).toHaveProperty('messages', [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Color?' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AQ==' } },
        ],
      },
    ])
  })

  it('advertises every built-in WebLLM vision model', () => {
    const ids = prebuiltAppConfig.model_list
      .filter((record) => record.model_type === ModelType.VLM)
      .map((record) => record.model_id)

    for (const modelId of ids) {
      const { model } = setup(undefined, { modelId })
      expect(model.input).toEqual(['text', 'image'])
    }
  })

  it('allows callers to disable built-in vision input', () => {
    const { model } = setup(undefined, {
      modelId: 'Phi-3.5-vision-instruct-q4f16_1-MLC',
      imageInputs: false,
    })

    expect(model.input).toEqual(['text'])
  })
})

function reverseKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(reverseKeys)
  if (typeof value !== 'object' || value === null) return value
  return Object.fromEntries(
    Object.entries(value)
      .reverse()
      .map(([key, item]) => [key, reverseKeys(item)]),
  )
}
