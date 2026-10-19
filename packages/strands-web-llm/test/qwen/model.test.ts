import { Agent, FunctionTool, Message, TextBlock } from '@strands-agents/sdk'
import { expect, it, vi } from 'vitest'
import { chunk, chunks, collect, setup } from '../../fixtures/engine.js'

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
  model.updateConfig({ modelId, temperature: 0, maxTokens: 128 })
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

  expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'maxTokens' })
  expect(events).toContainEqual({
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

  expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
  expect(events).toContainEqual({
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
