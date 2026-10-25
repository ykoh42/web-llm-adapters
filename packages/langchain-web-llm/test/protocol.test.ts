import { describe, expect, it, vi } from 'vitest'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import { ChatWebLLM } from '../src/index.js'
import { tool } from '@langchain/core/tools'
import { chunk, chunks, setup } from '../fixtures/engine.js'

const definition = {
  type: 'function',
  function: {
    name: 'lookup',
    description: 'Look up a code.',
    parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] },
  },
}
const qwen = 'Qwen3-8B-q4f16_1-MLC'
const gemmaId = 'gemma-4-E2B-it-q4f16_1-MLC'
function raw(text: string, finish: 'stop' | 'length' = 'stop') {
  return {
    id: 'raw',
    object: 'text_completion' as const,
    created: 0,
    model: gemmaId,
    choices: [{ index: 0, text, finish_reason: finish, logprobs: null }],
  }
}

describe('model protocols', () => {
  it('constrains verified Qwen tool arguments on request and allows disabling it per call', async () => {
    const output = '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
    const { engine, create } = setup(chunk({ content: output }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: qwen, request: { constrainedToolCalls: true } })

    const result = await model.bindTools([definition]).invoke('Find A7')
    await model
      .bindTools([definition])
      .invoke('Find A7', { request: { constrainedToolCalls: false } })

    expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expect(create.mock.calls[0]?.[0].response_format?.type).toBe('structural_tag')
    expect(create.mock.calls[1]?.[0]).not.toHaveProperty('response_format')
    expect(create.mock.calls[1]?.[0]).not.toHaveProperty('constrainedToolCalls')
  })

  it('keeps encoded references unchanged before LangChain tool validation', async () => {
    const $ref = '#/properties/code/definitions/value%7E1text'
    const callback = vi.fn()
    const lookup = tool(callback, {
      name: 'lookup',
      description: 'Look up a code.',
      schema: {
        type: 'object',
        properties: {
          code: { $ref, definitions: { 'value/text': { type: 'string' } } },
        },
        required: ['code'],
      },
    })
    const { engine, create } = setup(
      chunk(
        {
          content:
            '<tool_call><function=lookup><parameter=code>123</parameter></function></tool_call>',
        },
        'stop',
      ),
    )
    const model = new ChatWebLLM({ engine, modelId: 'Qwen3.5-4B-q4f16_1-MLC' })
    const response = await model.bindTools([lookup]).invoke('Find code 123.')
    expect(response.tool_calls).toMatchObject([{ name: 'lookup', args: { code: '123' } }])
    expect(create.mock.calls[0]?.[0].messages[0]?.content).toContain($ref)
    await expect(lookup.invoke({ code: '123' })).rejects.toThrow('Unresolved $ref')
    expect(callback).not.toHaveBeenCalled()
  })

  it('preserves a truncated Gemma tool call as text without exposing it as a tool', async () => {
    const source = '<|tool_call>call:lookup{code:<|"|>A7'
    const { engine } = setup()
    vi.spyOn(engine.completions, 'create').mockResolvedValue(
      (async function* () {
        yield raw(source, 'length')
      })(),
    )
    const model = new ChatWebLLM({ engine, modelId: gemmaId })

    const result = await model.bindTools([definition]).invoke('Find A7')

    expect(result.text).toBe(source)
    expect(result.tool_calls).toEqual([])
    expect(result.response_metadata.finish_reason).toBe('length')
  })

  it.each(['Qwen2.5-3B-Instruct-q4f16_1-MLC', qwen, 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s keeps tool syntax as text when no tools are bound',
    async (modelId) => {
      const thinking = !modelId.startsWith('Qwen2.5-')
      const reply = 'Example: <tool_call>{"name":"lookup","arguments":{}}</tool_call>'
      const text = thinking ? reply : '<think>Literal text.</think>' + reply
      const source = (thinking ? '<think>Plan.</think>' : '') + text
      const parts = source.match(/[\s\S]{1,3}/g)!
      const { engine } = setup(
        ...parts.map((content, index) =>
          chunk({ content }, index === parts.length - 1 ? 'stop' : null),
        ),
      )
      const model = new ChatWebLLM({ engine, modelId })

      const result = await model.invoke('Show the tool syntax.')

      expect(result.text).toBe(text)
      expect(result.tool_calls).toEqual([])
      expect(result.contentBlocks.filter((block) => block.type === 'reasoning')).toEqual(
        thinking ? [expect.objectContaining({ reasoning: 'Plan.' })] : [],
      )
    },
  )

  it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s preserves text and thinking order regardless of chunk size',
    async (modelId) => {
      const source = 'Intro<think>First</think>Middle<think>Second</think>End'
      for (const size of [1, 3, source.length]) {
        const parts = Array.from({ length: Math.ceil(source.length / size) }, (_, index) =>
          source.slice(index * size, (index + 1) * size),
        )
        const { engine } = setup(
          ...parts.map((content, index) =>
            chunk({ content }, index === parts.length - 1 ? 'stop' : null),
          ),
        )
        const model = new ChatWebLLM({ engine, modelId })

        const result = await model.invoke('Continue')

        expect(result.contentBlocks).toEqual([
          expect.objectContaining({ type: 'text', text: 'Intro' }),
          expect.objectContaining({ type: 'reasoning', reasoning: 'First' }),
          expect.objectContaining({ type: 'text', text: 'Middle' }),
          expect.objectContaining({ type: 'reasoning', reasoning: 'Second' }),
          expect.objectContaining({ type: 'text', text: 'End' }),
        ])
      }
    },
  )

  it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s preserves text and thinking after tool calls on stop or truncation',
    async (modelId) => {
      const call = modelId.startsWith('Qwen3.5-')
        ? '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
        : '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
      const source = call + 'After<think>Later</think>End'
      for (const finish of ['stop', 'length'] as const) {
        for (const size of [1, 3, source.length]) {
          const parts = Array.from({ length: Math.ceil(source.length / size) }, (_, index) =>
            source.slice(index * size, (index + 1) * size),
          )
          const { engine } = setup(
            ...parts.map((content, index) =>
              chunk({ content }, index === parts.length - 1 ? finish : null),
            ),
          )
          const model = new ChatWebLLM({ engine, modelId })

          const result = await model.bindTools([definition]).invoke('Continue')

          expect(result.contentBlocks.filter((block) => block.type !== 'tool_call')).toEqual([
            expect.objectContaining({
              type: 'text',
              text: (finish === 'length' ? call : '') + 'After',
            }),
            expect.objectContaining({ type: 'reasoning', reasoning: 'Later' }),
            expect.objectContaining({ type: 'text', text: 'End' }),
          ])
          expect(result.tool_calls).toHaveLength(finish === 'stop' ? 1 : 0)
        }
      }
    },
  )

  it.each(['test-model', gemmaId, qwen, 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s preserves empty assistant history as text WebLLM can use',
    async (modelId) => {
      const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
      const model = new ChatWebLLM({ engine, modelId })
      await model.invoke([new HumanMessage('Hi'), new AIMessage(''), new HumanMessage('Continue')])

      expect(create.mock.calls[0]?.[0].messages[1]).toEqual({ role: 'assistant', content: '' })
    },
  )

  it('uses the canonical Gemma prompt, thoughts and named-tool prefix', async () => {
    const { engine, create } = setup()
    const complete = vi.spyOn(engine.completions, 'create').mockResolvedValue(
      (async function* () {
        yield raw('code:<|"|>A7<|"|>}<tool_call|>')
      })(),
    )
    const model = new ChatWebLLM({
      engine,
      modelId: gemmaId,
      request: { extra_body: { enable_thinking: true } },
    })
    const result = await model.bindTools([definition], { tool_choice: 'lookup' }).invoke('Find A7.')
    expect(result.tool_calls?.[0]).toMatchObject({ name: 'lookup', args: { code: 'A7' } })
    expect(complete.mock.calls[0]?.[0]).toMatchObject({
      prompt: expect.stringMatching(/^<\|turn>system\n<\|think\|>\n/),
    })
    expect(complete.mock.calls[0]?.[0].prompt).toContain('<|tool>declaration:lookup')
    expect(complete.mock.calls[0]?.[0].prompt.endsWith('<|tool_call>call:lookup{')).toBe(true)
    expect(create).not.toHaveBeenCalled()
  })

  it('keeps Gemma reasoning for a tool reply and drops it after the next user turn', async () => {
    const { engine } = setup()
    const complete = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield raw('<channel|>42')
      })(),
    )
    const model = new ChatWebLLM({
      engine,
      modelId: gemmaId,
      request: { extra_body: { enable_thinking: true } },
    })
    const history = [
      new HumanMessage('Find A7'),
      new AIMessage({
        content: [{ type: 'reasoning', reasoning: 'Look up A7' }],
        tool_calls: [{ id: 'call', name: 'lookup', args: { code: 'A7' } }],
      }),
      new ToolMessage({ content: '42', tool_call_id: 'call' }),
    ]
    await model.bindTools([definition]).invoke(history)
    expect(complete.mock.calls[0]?.[0].prompt).toContain('<|channel>thought\nLook up A7')
    complete.mockResolvedValueOnce(
      (async function* () {
        yield raw('<|channel>thought\nUse the history.<channel|>42')
      })(),
    )
    await model.bindTools([definition]).invoke([...history, new HumanMessage('Repeat the answer')])
    expect(complete.mock.calls[1]?.[0].prompt).not.toContain('Look up A7')
    expect(history[1]?.content).toContainEqual({ type: 'reasoning', reasoning: 'Look up A7' })
  })

  it.each([
    { response_format: { type: 'json_object' as const } },
    { response_format: { schema: '{}' } },
    { response_format: { type: 'text' as const, schema: '{}' } },
    { response_format: { type: 'text' as const, grammar: 'root ::= "x"' } },
    { response_format: { type: 'text' as const, structural_tag: '{}' } },
    { extra_body: { resumable: { enabled: true } } },
    { ignore_eos: true },
  ])('rejects incompatible Gemma raw options %j', async (request) => {
    const { engine, create } = setup()
    const complete = vi.spyOn(engine.completions, 'create')
    const model = new ChatWebLLM({ engine, modelId: gemmaId, request })
    await expect(model.bindTools([definition]).invoke('Hi')).rejects.toThrow('does not support')
    expect(create).not.toHaveBeenCalled()
    expect(complete).not.toHaveBeenCalled()
  })

  it.each([{}, { type: 'text' as const }])(
    'allows plain Gemma response format %j in protocol mode',
    async (response_format) => {
      const { engine } = setup()
      const complete = vi.spyOn(engine.completions, 'create').mockResolvedValue(
        (async function* () {
          yield raw('Done')
        })(),
      )
      const model = new ChatWebLLM({ engine, modelId: gemmaId, request: { response_format } })

      expect((await model.bindTools([definition]).invoke('Hi')).text).toBe('Done')
      expect(complete.mock.calls[0]?.[0]).not.toHaveProperty('response_format')
    },
  )

  it('allows automatic Qwen3 1.7B chat when no tools are registered', async () => {
    const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: 'Qwen3-1.7B-q4f16_1-MLC' })

    expect((await model.bindTools([], { tool_choice: 'auto' }).invoke('Hi')).text).toBe('Done')
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty('tools')
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty('tool_choice')
  })

  it('does not send WebLLM’s Qwen-specific false prefix to Gemma chat', async () => {
    const { engine, create } = setup(chunk({ content: 'Hi' }, 'stop'))
    const model = new ChatWebLLM({
      engine,
      modelId: gemmaId,
      request: { extra_body: { enable_thinking: false } },
    })
    expect((await model.invoke('Hi')).text).toBe('Hi')
    expect(create.mock.calls[0]?.[0].extra_body).not.toHaveProperty('enable_thinking')
  })

  it.each(['test-model', gemmaId])('preserves native message names for %s', async (modelId) => {
    const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId })

    await model.invoke([
      new HumanMessage({ content: 'Hi', name: 'Ada' }),
      new AIMessage({ content: 'Hello', name: 'Assistant' }),
      new HumanMessage('Continue'),
    ])

    expect(create.mock.calls[0]?.[0].messages).toEqual([
      { role: 'user', content: 'Hi', name: 'Ada' },
      { role: 'assistant', content: 'Hello', name: 'Assistant' },
      { role: 'user', content: 'Continue' },
    ])
  })

  it.each([qwen, 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s replays thinking with canonical boundaries and keeps tool replies together',
    async (modelId) => {
      const { engine, create } = setup(chunk({ content: '42' }, 'stop'))
      const model = new ChatWebLLM({ engine, modelId })
      await model.bindTools([definition]).invoke([
        new SystemMessage('Custom system'),
        new HumanMessage('Hi'),
        new AIMessage({
          content: [
            { type: 'reasoning', reasoning: '\n  Thought  \n' },
            { type: 'text', text: '\n\nChecking.  \n' },
          ],
          tool_calls: [
            { id: 'a', name: 'lookup', args: { code: 'A7' } },
            { id: 'b', name: 'lookup', args: { code: 'B8' } },
          ],
        }),
        new ToolMessage({ content: '42', tool_call_id: 'a' }),
        new ToolMessage({ content: '43', tool_call_id: 'b' }),
      ])
      const messages = create.mock.calls[0]?.[0].messages
      expect(messages?.[0]?.content).toContain('Custom system')
      expect(
        String(messages?.[2]?.content).startsWith(
          modelId === qwen
            ? '<think>\n  Thought  \n</think>\n\nChecking.  \n'
            : '<think>\nThought\n</think>\n\nChecking.\n\n<tool_call>',
        ),
      ).toBe(true)
      expect(messages?.at(-1)?.content).toBe(
        '<tool_response>\n42\n</tool_response>\n<tool_response>\n43\n</tool_response>',
      )
    },
  )

  it('omits LangGraph’s default model name from Qwen tool history', async () => {
    const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: qwen })

    await model.bindTools([definition]).invoke([
      new HumanMessage('Find A7'),
      new AIMessage({
        name: 'model',
        content: '',
        tool_calls: [{ id: 'call', name: 'lookup', args: { code: 'A7' } }],
      }),
      new ToolMessage({ name: 'lookup', content: '42', tool_call_id: 'call' }),
    ])

    const messages = create.mock.calls[0]?.[0].messages
    expect(messages?.some((message) => 'name' in message && message.name === 'model')).toBe(false)
    expect(messages?.at(-1)?.content).toContain('<tool_response>\n42\n</tool_response>')
  })

  it('preserves user and assistant names in Qwen history', async () => {
    const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: qwen })

    await model.invoke([
      new HumanMessage({ content: 'Hi', name: 'Ada' }),
      new AIMessage({ content: 'Hello', name: 'Assistant' }),
      new HumanMessage('Continue'),
    ])

    expect(create.mock.calls[0]?.[0].messages).toContainEqual({
      role: 'user',
      content: 'Hi',
      name: 'Ada',
    })
    expect(create.mock.calls[0]?.[0].messages).toContainEqual({
      role: 'assistant',
      content: 'Hello',
      name: 'Assistant',
    })
  })

  it('rejects named messages when Gemma protocol mode cannot preserve them', async () => {
    const { engine, create } = setup()
    const complete = vi.spyOn(engine.completions, 'create')
    const model = new ChatWebLLM({ engine, modelId: gemmaId })

    await expect(
      model.bindTools([definition]).invoke([new HumanMessage({ content: 'Hi', name: 'Ada' })]),
    ).rejects.toThrow('cannot preserve message names')
    await expect(
      model
        .bindTools([definition])
        .invoke([
          new AIMessage({ content: 'Hello', name: 'Assistant' }),
          new HumanMessage('Continue'),
        ]),
    ).rejects.toThrow('cannot preserve message names')
    expect(create).not.toHaveBeenCalled()
    expect(complete).not.toHaveBeenCalled()
  })

  it('uses the call ID for LangGraph tool names in Gemma history', async () => {
    const { engine } = setup()
    const complete = vi.spyOn(engine.completions, 'create').mockResolvedValue(
      (async function* () {
        yield raw('42')
      })(),
    )
    const model = new ChatWebLLM({ engine, modelId: gemmaId })

    await model.bindTools([definition]).invoke([
      new HumanMessage('Find A7'),
      new AIMessage({
        name: 'model',
        content: '',
        tool_calls: [{ id: 'call', name: 'lookup', args: { code: 'A7' } }],
      }),
      new ToolMessage({ name: 'lookup', content: '42', tool_call_id: 'call' }),
    ])

    expect(complete.mock.calls[0]?.[0].prompt).toContain('<|tool_response>response:lookup{value:')
    expect(complete.mock.calls[0]?.[0].prompt).toContain('42<|"|>}<tool_response|>')
  })

  it('keeps literal tool tags in native constrained text instead of parsing them', async () => {
    const { engine } = setup(
      chunk({ content: '{"text":"<tool_call>literal</tool_call>"}' }, 'stop'),
    )
    const model = new ChatWebLLM({
      engine,
      modelId: qwen,
      request: { response_format: { type: 'json_object' } },
    })
    expect((await model.invoke('Hi')).text).toBe('{"text":"<tool_call>literal</tool_call>"}')
    await expect(model.bindTools([definition]).invoke('Hi')).rejects.toThrow('structured response')
  })

  it('uses the exact Qwen2.5 default and respects an explicitly empty system message', async () => {
    const { engine, create } = setup(chunk({ content: 'Hi' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC' })
    await model.invoke('Hi')
    expect(create.mock.calls[0]?.[0].messages[0]).toEqual({
      role: 'system',
      content: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
    })
    await model.invoke([new SystemMessage(''), new HumanMessage('Hi')])
    expect(create.mock.calls[1]?.[0].messages[0]).toEqual({ role: 'system', content: '' })
  })

  it('rejects unknown Qwen tool choices', async () => {
    const { engine } = setup()
    const supported = new ChatWebLLM({ engine, modelId: qwen })
    await expect(
      supported.bindTools([definition], { tool_choice: { type: 'bad' } }).invoke('Hi'),
    ).rejects.toThrow('tool_choice')
    await expect(
      supported
        .bindTools([definition], {
          tool_choice: { type: 'function', function: { name: 'missing' } },
        })
        .invoke('Hi'),
    ).rejects.toThrow('not registered')
  })

  it.each([qwen, 'Qwen3.5-4B-q4f16_1-MLC'])(
    'rejects a required Qwen tool choice before generating without registered tools: %s',
    async (modelId) => {
      const { engine, create } = setup()
      const model = new ChatWebLLM({ engine, modelId })

      await expect(model.bindTools([], { tool_choice: 'required' }).invoke('Hi')).rejects.toThrow(
        'registered tool',
      )
      await expect(model.bindTools([], { tool_choice: 'missing' }).invoke('Hi')).rejects.toThrow(
        'registered tool',
      )
      expect(create).not.toHaveBeenCalled()
    },
  )

  it('keeps interleaved native calls ordered without changing their argument bytes', async () => {
    const native = [
      chunk({
        tool_calls: [
          { index: 7, id: 'second', function: { name: 'lo', arguments: ' { "code": ' } },
          { index: 2, function: { name: 'look', arguments: '{ "code": "A' } },
        ],
      }),
      chunk(
        {
          tool_calls: [
            { index: 2, id: 'first', function: { name: 'up', arguments: '7" }' } },
            { index: 7, function: { name: 'okup', arguments: '"A8" } ' } },
          ],
        },
        'tool_calls',
      ),
    ]
    const original = structuredClone(native)
    const { model } = setup(...native)

    const result = await model.bindTools([definition]).invoke('Look up both codes.')

    expect(result.tool_calls).toEqual([
      { name: 'lookup', args: { code: 'A7' }, id: 'first', type: 'tool_call' },
      { name: 'lookup', args: { code: 'A8' }, id: 'second', type: 'tool_call' },
    ])
    expect(result.tool_call_chunks?.map(({ index, args }) => ({ index, args }))).toEqual([
      { index: 2, args: '{ "code": "A7" }' },
      { index: 7, args: ' { "code": "A8" } ' },
    ])
    expect(native).toEqual(original)
  })

  it('preserves native tool fragments and rejects invalid tool JSON', async () => {
    const { model, create } = setup(
      chunk({
        tool_calls: [
          {
            index: 0,
            id: 'id',
            type: 'function',
            function: { name: 'look', arguments: '{"code":' },
          },
        ],
      }),
      chunk(
        {
          tool_calls: [
            { index: 0, type: 'function', function: { name: 'up', arguments: '"A7"}' } },
          ],
        },
        'tool_calls',
      ),
    )
    const result = await model.bindTools([definition]).invoke('Hi')
    expect(result.tool_calls).toEqual([
      { name: 'lookup', args: { code: 'A7' }, id: 'id', type: 'tool_call' },
    ])
    create.mockResolvedValueOnce(chunks(chunk({}, 'tool_calls')))
    await expect(model.bindTools([definition]).invoke('Hi')).rejects.toThrow('no tool calls')
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            tool_calls: [
              { index: 0, type: 'function', function: { name: 'lookup', arguments: '[]' } },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    await expect(model.bindTools([definition]).invoke('Hi')).rejects.toThrow('must be an object')
  })
})

it('preserves a thought stopped by a requested native stop string', async () => {
  const { engine } = setup()
  const complete = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
    (async function* () {
      yield raw('<|channel>thought\nA thought before the stop')
    })(),
  )
  const model = new ChatWebLLM({
    engine,
    modelId: gemmaId,
    request: { extra_body: { enable_thinking: true } },
  })
  const stopped = await model.invoke('Hi', { stop: ['you'] })
  expect(stopped.text).toBe('')
  expect(stopped.contentBlocks).toContainEqual(
    expect.objectContaining({ type: 'reasoning', reasoning: 'A thought before the stop' }),
  )
  expect(stopped.response_metadata.finish_reason).toBe('stop')
  expect(complete.mock.calls[0]?.[0].stop).toEqual(['you'])
  await expect(model.invoke('Hi')).rejects.toThrow('inside a thinking block')
})

it('rejects a model that violates required or named tool choice', async () => {
  const { engine, create } = setup(chunk({ content: 'No call' }, 'stop'))
  const model = new ChatWebLLM({ engine, modelId: qwen })
  await expect(
    model.bindTools([definition], { tool_choice: 'required' }).invoke('Hi'),
  ).rejects.toThrow('required tool')
  create.mockResolvedValueOnce(
    chunks(
      chunk({ content: '<tool_call>\n{"name":"other","arguments":{}}\n</tool_call>' }, 'stop'),
    ),
  )
  await expect(
    model
      .bindTools(
        [
          definition,
          {
            type: 'function',
            function: {
              name: 'other',
              description: 'Other',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
        { tool_choice: 'lookup' },
      )
      .invoke('Hi'),
  ).rejects.toThrow('outside tool_choice')
})
