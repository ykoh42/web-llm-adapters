import { Agent, FunctionTool, Message, TextBlock, type ModelStreamEvent } from '@strands-agents/sdk'
import { expect, it, vi } from 'vitest'
import { chunk, chunks, collect, setup } from '../../fixtures/engine.js'
import { WebLLMModel } from '../../src/index.js'

it.each(['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  '%s reports an async chunk callback failure and releases the engine',
  async (modelId) => {
    const { engine, model: next, create } = setup(chunk({ content: 'Hi' }, 'stop'))
    const onChunk = vi.fn().mockImplementationOnce(async () => {
      await Promise.resolve()
      throw new Error('Async callback failed')
    })
    const model = new WebLLMModel({ engine, modelId, onChunk })
    const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

    await expect(collect(model.stream(messages))).rejects.toThrow('Async callback failed')
    expect(onChunk).toHaveBeenCalledOnce()
    await expect(collect(next.stream(messages))).resolves.toContainEqual(
      expect.objectContaining({ type: 'modelMessageStopEvent' }),
    )
    expect(create).toHaveBeenCalledTimes(2)
  },
)

it.each(['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  '%s keeps tool syntax as text when no tools are registered',
  async (modelId) => {
    const thinking = !modelId.startsWith('Qwen2.5-')
    const reply = 'Example: <tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    const text = thinking ? reply : '<think>Literal text.</think>' + reply
    const source = (thinking ? '<think>Plan.</think>' : '') + text
    const parts = source.match(/[\s\S]{1,3}/g)!
    const { model, create } = setup()
    model.updateConfig({ modelId })
    create.mockResolvedValueOnce(
      chunks(
        ...parts.map((content, index) =>
          chunk({ content }, index === parts.length - 1 ? 'stop' : null),
        ),
      ),
    )
    const agent = new Agent({ model, printer: false })

    const result = await agent.invoke('Show the tool syntax.')

    expect(
      result.lastMessage.content
        .filter((block) => block.type === 'textBlock')
        .map((block) => block.text)
        .join(''),
    ).toBe(text)
    expect(result.lastMessage.content.filter((block) => block.type === 'toolUseBlock')).toEqual([])
    expect(result.lastMessage.content.filter((block) => block.type === 'reasoningBlock')).toEqual(
      thinking ? [expect.objectContaining({ text: 'Plan.' })] : [],
    )
  },
)

it.each([
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
])('enables constrained tool arguments for verified model %s when requested', async (modelId) => {
  const { model, create } = setup(chunk({ content: 'Ready' }, 'stop'))
  model.updateConfig({ modelId, request: { constrainedToolCalls: true } })

  await collect(
    model.stream([new Message({ role: 'user', content: [new TextBlock('Use lookup.')] })], {
      toolSpecs: [{ name: 'lookup', description: 'Look up a value.' }],
    }),
  )

  expect(create.mock.calls[0]?.[0].response_format?.type).toBe('structural_tag')
})

it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  '%s preserves text and thinking order regardless of chunk size',
  async (modelId) => {
    const source = 'Intro<think>First</think>Middle<think>Second</think>End'
    for (const size of [1, 3, source.length]) {
      const parts = Array.from({ length: Math.ceil(source.length / size) }, (_, index) =>
        source.slice(index * size, (index + 1) * size),
      )
      const { model } = setup(
        ...parts.map((content, index) =>
          chunk({ content }, index === parts.length - 1 ? 'stop' : null),
        ),
      )
      model.updateConfig({ modelId })
      const agent = new Agent({ model, printer: false })

      const result = await agent.invoke('Continue')

      expect(result.lastMessage.content).toEqual([
        expect.objectContaining({ type: 'textBlock', text: 'Intro' }),
        expect.objectContaining({ type: 'reasoningBlock', text: 'First' }),
        expect.objectContaining({ type: 'textBlock', text: 'Middle' }),
        expect.objectContaining({ type: 'reasoningBlock', text: 'Second' }),
        expect.objectContaining({ type: 'textBlock', text: 'End' }),
      ])
    }
  },
)

it.each([
  {
    modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    call: '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>',
  },
  {
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    call: '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>',
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    call: '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>',
  },
])('$modelId keeps text and tools in model order', async ({ modelId, call }) => {
  const source = `Before${call}Between${call}After`
  for (const size of [1, source.length]) {
    const parts = source.match(new RegExp(`[\\s\\S]{1,${size}}`, 'g'))!
    const { model } = setup(
      ...parts.map((content, index) =>
        chunk({ content }, index === parts.length - 1 ? 'stop' : null),
      ),
    )
    model.updateConfig({ modelId })
    const events = await collect(
      model.stream([new Message({ role: 'user', content: [new TextBlock('Look up A7.')] })], {
        toolSpecs: [{ name: 'lookup', description: 'Look up a code.' }],
      }),
    )
    const blocks = events.flatMap((event) => {
      if (event.type !== 'modelContentBlockStartEvent') return []
      return [event.start?.type === 'toolUseStart' ? 'tool' : 'text']
    })
    expect(blocks, `chunk size ${size}`).toEqual(['text', 'tool', 'text', 'tool', 'text'])
    expect(
      events
        .flatMap((event) =>
          event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta'
            ? [event.delta.text]
            : [],
        )
        .join(''),
    ).toBe('BeforeBetweenAfter')
    expect(events.at(-1)).toMatchObject({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
  }
})

it.each([
  {
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    call: '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>',
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    call: '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>',
  },
])(
  '$modelId keeps later text without publishing tools when generation ends early',
  async ({ modelId, call }) => {
    for (const finish of ['length', 'abort', 'error'] as const) {
      const failure = new Error('Stream failed')
      const { model, create } = setup()
      model.updateConfig({ modelId })
      create.mockResolvedValueOnce(
        (async function* () {
          yield chunk({ content: `Before${call}After` }, finish === 'error' ? null : finish)
          if (finish === 'error') throw failure
        })(),
      )
      const events: ModelStreamEvent[] = []
      const pending = (async () => {
        for await (const event of model.stream(
          [new Message({ role: 'user', content: [new TextBlock('Look up A7.')] })],
          { toolSpecs: [{ name: 'lookup', description: 'Look up a code.' }] },
        ))
          events.push(event)
      })()

      if (finish === 'error') await expect(pending).rejects.toBe(failure)
      else await pending
      expect(
        events
          .flatMap((event) =>
            event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta'
              ? [event.delta.text]
              : [],
          )
          .join(''),
      ).toBe('BeforeAfter')
      expect(events).not.toContainEqual(
        expect.objectContaining({
          type: 'modelContentBlockStartEvent',
          start: expect.objectContaining({ type: 'toolUseStart' }),
        }),
      )
      if (finish !== 'error')
        expect(events.at(-1)).toMatchObject({
          type: 'modelMessageStopEvent',
          stopReason: finish === 'length' ? 'maxTokens' : 'cancelled',
        })
    }
  },
)

it('rejects a required tool choice before generating without registered tools', async () => {
  const { model, create } = setup()
  model.updateConfig({ modelId: 'Qwen3-4B-q4f16_1-MLC' })
  const messages = [new Message({ role: 'user', content: [new TextBlock('Hi.')] })]

  await expect(collect(model.stream(messages, { toolChoice: { any: {} } }))).rejects.toThrow(
    'registered tool',
  )
  await expect(
    collect(model.stream(messages, { toolChoice: { tool: { name: 'lookup' } } })),
  ).rejects.toThrow('registered tool')
  expect(create).not.toHaveBeenCalled()
})

it('runs a Qwen3 tool through a Strands Agent and continues from its result', async () => {
  const { model, create } = setup()
  const modelId = 'Qwen3-4B-q4f16_1-MLC'
  model.updateConfig({
    modelId,
    temperature: 0,
    maxTokens: 128,
    request: { constrainedToolCalls: true },
  })
  const input = vi.fn((value: unknown) => {
    expect(value).toEqual({ city: 'Seoul' })
    return { code: 'pineapple-42' }
  })
  const lookup = new FunctionTool({
    name: 'lookup',
    description: 'Return the verification code for a city.',
    inputSchema: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
    callback: input,
  })
  create
    .mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Find the code.</think><tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}</tool_call>',
          },
          'stop',
        ),
      ),
    )
    .mockResolvedValueOnce(
      chunks(
        chunk(
          { content: '<think>Use the returned code.</think>The code is pineapple-42.' },
          'stop',
        ),
      ),
    )
  const agent = new Agent({ model, tools: [lookup], printer: false })

  const result = await agent.invoke('Use lookup to find the code for Seoul.')

  expect(input.mock.calls[0]?.[0]).toEqual({ city: 'Seoul' })
  expect(result.stopReason).toBe('endTurn')
  expect(String(result)).toContain('pineapple-42')
  expect(create).toHaveBeenCalledTimes(2)
  const firstRequest = create.mock.calls[0]![0]
  expect(firstRequest).not.toHaveProperty('tools')
  expect(firstRequest.messages[0]).toMatchObject({ role: 'system' })
  expect(firstRequest.messages[0]?.content).toContain('<tools>')
  expect(firstRequest.messages[0]?.content).toContain(
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>',
  )
  expect(firstRequest.messages[0]?.content).not.toContain('Wait for the tool result')
  expect(firstRequest.messages[0]?.content).not.toContain('Do not repeat')
  expect(firstRequest.response_format?.type).toBe('structural_tag')
  expect(JSON.stringify(firstRequest.response_format)).toContain('lookup')
  expect(create.mock.calls[1]![0].messages[0]?.content).not.toContain('Do not repeat')
  const nextMessages = create.mock.calls[1]![0].messages
  expect(nextMessages).toContainEqual(
    expect.objectContaining({
      role: 'assistant',
      content: expect.stringContaining(
        '<tool_call>\n{"name": "lookup", "arguments": {"city":"Seoul"}}\n</tool_call>',
      ),
    }),
  )
  expect(nextMessages).toContainEqual(
    expect.objectContaining({ role: 'user', content: expect.stringContaining('pineapple-42') }),
  )
})

it('can disable constrained tool arguments for a Qwen request', async () => {
  const { model, create } = setup(
    chunk(
      {
        content: '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}</tool_call>',
      },
      'stop',
    ),
  )
  model.updateConfig({
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    request: { constrainedToolCalls: false },
  })

  await collect(
    model.stream([new Message({ role: 'user', content: [new TextBlock('Look up Seoul.')] })], {
      toolSpecs: [{ name: 'lookup', description: 'Look up a city.' }],
    }),
  )

  expect(create.mock.calls[0]?.[0]).not.toHaveProperty('constrainedToolCalls')
  expect(create.mock.calls[0]?.[0]).not.toHaveProperty('response_format')
})

it('rejects constrained tool arguments for an unverified Qwen protocol', async () => {
  const { model, create } = setup()
  model.updateConfig({
    modelId: 'Qwen3.5-4B-q4f32_1-MLC',
    request: { constrainedToolCalls: true },
  })

  await expect(
    collect(
      model.stream([new Message({ role: 'user', content: [new TextBlock('Look up Seoul.')] })], {
        toolSpecs: [{ name: 'lookup', description: 'Look up a city.' }],
      }),
    ),
  ).rejects.toThrow('Constrained tool calls are not verified')
  expect(create).not.toHaveBeenCalled()
})

it('keeps Qwen3 thinking separate during ordinary chat', async () => {
  const { model, create } = setup()
  const modelId = 'Qwen3-4B-q4f16_1-MLC'
  model.updateConfig({ modelId })
  create.mockResolvedValueOnce(
    chunks(chunk({ content: '<think>Plan first.</think>Done.' }, 'stop')),
  )

  const events = await collect(
    model.stream([new Message({ role: 'user', content: [new TextBlock('Say done.')] })]),
  )

  expect(events).toContainEqual({
    type: 'modelContentBlockDeltaEvent',
    delta: { type: 'reasoningContentDelta', text: 'Plan first.' },
  })
  expect(events).toContainEqual({
    type: 'modelContentBlockDeltaEvent',
    delta: { type: 'textDelta', text: 'Done.' },
  })
  expect(create.mock.calls[0]?.[0]).not.toHaveProperty('tools')
})

it('preserves protocol-like strings in constrained JSON output', async () => {
  const { model, create } = setup()
  const response = '{"text":"<think>hello</think>","tag":"<tool_call>"}'
  model.updateConfig({
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    request: { response_format: { type: 'json_object' }, extra_body: { enable_thinking: false } },
  })
  create.mockResolvedValueOnce(chunks(chunk({ content: response }, 'stop')))

  const events = await collect(
    model.stream([new Message({ role: 'user', content: [new TextBlock('Return JSON.')] })]),
  )

  expect(
    events
      .flatMap((event) =>
        event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta'
          ? [event.delta.text]
          : [],
      )
      .join(''),
  ).toBe(response)
  expect(
    events.some(
      (event) =>
        event.type === 'modelContentBlockDeltaEvent' &&
        event.delta.type === 'reasoningContentDelta',
    ),
  ).toBe(false)
})

it('rejects custom tool calls with constrained JSON output', async () => {
  const { model, create } = setup()
  model.updateConfig({
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    request: { response_format: { type: 'json_object' } },
  })

  await expect(
    collect(
      model.stream([new Message({ role: 'user', content: [new TextBlock('Call lookup.')] })], {
        toolSpecs: [{ name: 'lookup', description: 'Look up a city.' }],
      }),
    ),
  ).rejects.toThrow('Qwen tool calls cannot use a structured response format.')
  expect(create).not.toHaveBeenCalled()
})

it('preserves the token limit when a tool call is truncated', async () => {
  const { model, create } = setup()
  model.updateConfig({ modelId: 'Qwen3-4B-q4f16_1-MLC' })
  create.mockResolvedValueOnce(
    chunks(chunk({ content: '<tool_call>{"name":"lookup","arguments":{"city":"Se' }, 'length')),
  )

  const events = await collect(
    model.stream([new Message({ role: 'user', content: [new TextBlock('Look up a city.')] })], {
      toolSpecs: [{ name: 'lookup', description: 'Look up a city.' }],
    }),
  )

  expect(events).toContainEqual(
    expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'maxTokens' }),
  )
  expect(events).not.toContainEqual({
    type: 'modelContentBlockDeltaEvent',
    delta: { type: 'textDelta', text: '<tool_call>{"name":"lookup","arguments":{"city":"Se' },
  })
  expect(
    events.some(
      (event) =>
        event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart',
    ),
  ).toBe(false)
})

it('preserves cancellation when a tool call is interrupted', async () => {
  const { model, create } = setup()
  model.updateConfig({ modelId: 'Qwen3-4B-q4f16_1-MLC' })
  create.mockResolvedValueOnce(
    chunks(chunk({ content: '<tool_call>{"name":"lookup","arguments":{"city":"Se' }, 'abort')),
  )

  const events = await collect(
    model.stream([new Message({ role: 'user', content: [new TextBlock('Look up a city.')] })], {
      toolSpecs: [{ name: 'lookup', description: 'Look up a city.' }],
    }),
  )

  expect(events).toContainEqual(
    expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'cancelled' }),
  )
  expect(events).not.toContainEqual({
    type: 'modelContentBlockDeltaEvent',
    delta: { type: 'textDelta', text: '<tool_call>{"name":"lookup","arguments":{"city":"Se' },
  })
  expect(
    events.some(
      (event) =>
        event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart',
    ),
  ).toBe(false)
})
