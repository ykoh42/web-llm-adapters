import { describe, expect, it, vi } from 'vitest'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import { ChatWebLLM } from '../src/index.js'
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
    { extra_body: { resumable: { enabled: true } } },
    { ignore_eos: true },
  ])('rejects incompatible Gemma raw options %j', async (request) => {
    const { engine } = setup()
    const model = new ChatWebLLM({ engine, modelId: gemmaId, request })
    await expect(model.bindTools([definition]).invoke('Hi')).rejects.toThrow('does not support')
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

  it('sends named user and assistant messages through the native WebLLM protocol', async () => {
    const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: 'test-model' })

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

  it('keeps Qwen tool replies together and preserves current-turn reasoning', async () => {
    const { engine, create } = setup(chunk({ content: '42' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: qwen })
    await model.bindTools([definition]).invoke([
      new SystemMessage('Custom system'),
      new HumanMessage('Hi'),
      new AIMessage({
        content: [{ type: 'reasoning', reasoning: 'Thought' }],
        tool_calls: [
          { id: 'a', name: 'lookup', args: { code: 'A7' } },
          { id: 'b', name: 'lookup', args: { code: 'B8' } },
        ],
      }),
      new ToolMessage({ content: '42', tool_call_id: 'a' }),
      new ToolMessage({ content: '43', tool_call_id: 'b' }),
    ])
    const messages = create.mock.calls[0]?.[0].messages
    expect(messages?.[0]?.content).toMatch(/^Custom system\n\n# Tools/)
    expect(messages?.[2]?.content).toContain('<think>\nThought\n</think>')
    expect(messages?.at(-1)?.content).toBe(
      '<tool_response>\n42\n</tool_response>\n<tool_response>\n43\n</tool_response>',
    )
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

  it('rejects Qwen3 1.7B tool requests and unknown tool choices', async () => {
    const { engine } = setup()
    const model = new ChatWebLLM({ engine, modelId: 'Qwen3-1.7B-q4f16_1-MLC' })
    await expect(model.bindTools([definition]).invoke('Hi')).rejects.toThrow('not supported')
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
