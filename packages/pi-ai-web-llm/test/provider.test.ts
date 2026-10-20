import { describe, expect, it, vi } from 'vitest'
import { Type, type Context } from '@earendil-works/pi-ai'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { Agent } from '@earendil-works/pi-agent-core'
import { chunk, chunks, collect, setup } from './helpers.js'
import { createWebLLMProvider } from '../src/index.js'

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

  it('completes even when its events are not consumed', async () => {
    const { models, model } = setup()
    expect((await models.streamSimple(model, context).result()).stopReason).toBe('stop')
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

  it('buffers native tool deltas into a complete Pi tool call', async () => {
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
      'toolcall_end',
      'done',
    ])
    expect((await stream.result()).content).toEqual([
      { type: 'toolCall', id: 'call-1', name: 'lookup', arguments: { code: 'A7' } },
    ])
  })

  it('runs the real Pi Agent Core tool loop and replays the result to WebLLM', async () => {
    const { models, model, create } = setup()
    create
      .mockResolvedValueOnce(chunks(call()))
      .mockResolvedValueOnce(chunks(chunk({ content: 'ORCHID-42' }, 'stop')))
    const execute = vi.fn(async (id: string, args: unknown) => {
      expect(id).toBe('call-1')
      expect(args).toEqual({ code: 'A7' })
      return { content: [{ type: 'text' as const, text: 'ORCHID-42' }], details: {} }
    })
    const agent = new Agent({
      initialState: { model, tools: [{ ...lookup, label: 'Lookup', execute }] },
      streamFn: models.streamSimple.bind(models),
    })
    await agent.prompt('Look up A7.')
    expect(execute).toHaveBeenCalledTimes(1)
    expect(execute.mock.calls[0]?.[1]).toEqual({ code: 'A7' })
    expect(create.mock.calls[1]?.[0]).toMatchObject({
      messages: expect.arrayContaining([
        { role: 'tool', tool_call_id: 'call-1', content: 'ORCHID-42' },
      ]),
    })
    expect(agent.state.messages.at(-1)).toMatchObject({
      role: 'assistant',
      stopReason: 'stop',
      content: [{ type: 'text', text: 'ORCHID-42' }],
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

  it.each(['{', '[]', '{"code":1e999}', 'null'])(
    'does not publish executable tool calls for invalid arguments %s',
    async (args) => {
      const { models, model } = setup([call(args)])
      const stream = models.streamSimple(model, tools)
      expect((await collect(stream)).some((event) => event.type === 'toolcall_end')).toBe(false)
      expect((await stream.result()).stopReason).toBe('error')
    },
  )

  it('does not execute a tool the model was not offered', async () => {
    const { models, model } = setup([call('{}', 'unregistered')])
    expect(await models.completeSimple(model, tools)).toMatchObject({
      stopReason: 'error',
      errorMessage: expect.stringContaining('unknown tool'),
    })
  })

  it('enforces toolChoice none and withholds truncated calls', async () => {
    const { models, model, create } = setup([call()])
    expect((await models.completeSimple(model, tools, { toolChoice: 'none' })).stopReason).toBe(
      'error',
    )
    expect(create.mock.calls[0]?.[0]).toHaveProperty('tool_choice', 'none')
    create.mockResolvedValueOnce(
      chunks({ ...call('{'), choices: [{ ...call().choices[0]!, finish_reason: 'length' }] }),
    )
    const truncated = await models.completeSimple(model, tools)
    expect(truncated).toMatchObject({ stopReason: 'length', content: [] })
  })

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

  it('preserves base64 images on a vision model', async () => {
    const { models, model, create } = setup(undefined, { imageInputs: true })
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
})
