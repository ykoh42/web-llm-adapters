import { describe, expect, it, vi } from 'vitest'
import {
  Agent,
  FunctionTool,
  MaxTokensError,
  Message,
  TextBlock,
  ReasoningBlock,
} from '@strands-agents/sdk'
import type { ToolSpec } from '@strands-agents/sdk'
import { MLCEngine, type Completion } from '@mlc-ai/web-llm'
import { WebLLMModel } from '../src/index.js'
import { collect, pause, setup as chatSetup, chunk } from './helpers.js'

const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a value.' }]
const messages = [new Message({ role: 'user', content: [new TextBlock('Find x.')] })]
const call = '<|tool_call>call:lookup{key:<|"|>x<|"|>}<tool_call|>'

function part(
  text = '',
  finish: Completion['choices'][number]['finish_reason'] = null,
): Completion {
  return {
    id: 'test',
    object: 'text_completion',
    model: modelId,
    created: 0,
    choices: [{ text, index: 0, finish_reason: finish }],
  }
}
async function* parts(...values: Completion[]) {
  yield* values
}
function setup(...values: Completion[]) {
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  const create = vi.spyOn(engine.completions, 'create').mockResolvedValue(parts(...values))
  const chat = vi.spyOn(engine.chat.completions, 'create')
  const stop = vi.spyOn(engine, 'interruptGenerate').mockResolvedValue(undefined)
  const model = new WebLLMModel({ engine, modelId, maxTokens: 128, temperature: 0 })
  return { model, engine, create, chat, stop }
}
async function aggregate(model: WebLLMModel) {
  const stream = model.streamAggregated(messages, { toolSpecs: tools })
  for (;;) {
    const step = await stream.next()
    if (step.done) return step.value
  }
}

describe('Gemma tool model', () => {
  it('counts only thoughts kept in the current user turn without changing history', async () => {
    const { model } = setup()
    const history = [
      ...messages,
      new Message({
        role: 'assistant',
        content: [new ReasoningBlock({ text: 'word '.repeat(2400) }), new TextBlock('Old answer')],
      }),
      new Message({ role: 'user', content: [new TextBlock('Next question')] }),
    ]
    const clean = history.map(
      (message) =>
        new Message({
          role: message.role,
          content: message.content.filter((block) => block.type !== 'reasoningBlock'),
        }),
    )
    const snapshot = JSON.stringify(history)
    expect(await model.countTokens(history, { toolSpecs: tools })).toBe(
      await model.countTokens(clean, { toolSpecs: tools }),
    )
    expect(JSON.stringify(history)).toBe(snapshot)
    history.push(
      new Message({
        role: 'assistant',
        content: [new ReasoningBlock({ text: 'word '.repeat(2400) })],
      }),
    )
    for (const enable_thinking of [true, false]) {
      model.updateConfig({ request: { extra_body: { enable_thinking } } })
      expect(await model.countTokens(history)).toBeGreaterThan(
        (await model.countTokens(clean)) + 2900,
      )
    }
  })
  it('keeps tool results when native declarations fit the context window', async () => {
    const { model, create } = setup(part(call, 'stop'))
    create.mockResolvedValueOnce(parts(part(call, 'stop')))
    create.mockResolvedValueOnce(parts(part('secret-42', 'stop')))
    model.updateConfig({ contextWindowLimit: 4096, maxTokens: 128 })
    const tool = new FunctionTool({
      name: 'lookup',
      description: 'x'.repeat(8800),
      callback: () => 'secret-42',
    })
    const agent = new Agent({ model, tools: [tool], contextManager: 'auto', printer: false })
    expect(String(await agent.invoke('Find x.'))).toBe('secret-42')
    expect(create.mock.calls[1]![0].prompt).toContain(
      '<|tool_response>response:lookup{value:<|"|>secret-42<|"|>}',
    )
  })
  it('separates streamed thinking, preserves it through tools, and strips it on the next user turn', async () => {
    const { model, create, chat } = setup()
    model.updateConfig({ request: { extra_body: { enable_thinking: true } } })
    const thought = 'Check x. A quoted example is <|tool_call>call:lookup{}<tool_call|>.'
    const source = '<|channel>thought\n' + thought + '<channel|>' + call
    const lookup = vi.fn(() => 'secret-42')
    create.mockResolvedValueOnce(
      parts(...Array.from(source, (text) => part(text)), part('', 'stop')),
    )
    // The prompt after a tool response already opens the next thought channel.
    create.mockResolvedValueOnce(parts(part('Use the result.<channel|>secret-42', 'stop')))
    const agent = new Agent({
      model,
      printer: false,
      tools: [new FunctionTool({ ...tools[0]!, callback: lookup })],
    })
    await agent.invoke('Find x.')
    expect(lookup).toHaveBeenCalledTimes(1)
    expect(agent.messages[1]?.content).toEqual([
      new ReasoningBlock({ text: thought }),
      expect.objectContaining({ type: 'toolUseBlock', name: 'lookup' }),
    ])
    expect(agent.messages[3]?.content).toEqual([
      new ReasoningBlock({ text: 'Use the result.' }),
      new TextBlock('secret-42'),
    ])
    const prompt = create.mock.calls[1]![0].prompt
    expect(prompt).toContain('<|channel>thought\n' + thought + '<channel|>' + call)
    expect(prompt.endsWith('<tool_response|><|channel>thought\n')).toBe(true)
    expect(create.mock.calls[0]![0].extra_body).toBeUndefined()
    const snapshot = JSON.stringify(agent.messages)
    create.mockResolvedValueOnce(
      parts(part('<|channel>thought\nRecall.<channel|>secret-42', 'stop')),
    )
    expect((await agent.invoke('What did we find?')).lastMessage.content).toEqual([
      new ReasoningBlock({ text: 'Recall.' }),
      new TextBlock('secret-42'),
    ])
    const next = create.mock.calls[2]![0].prompt
    expect(next).not.toContain(thought)
    expect(next).not.toContain('Use the result.')
    expect(next).toContain('<tool_response|>secret-42<turn|>')
    expect(JSON.stringify(agent.messages.slice(0, 4))).toBe(snapshot)
    expect(chat).not.toHaveBeenCalled()
  })

  it('streams thinking before generation finishes and closes blocks when the channel changes', async () => {
    const { model, create } = setup()
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield part('Before<|channel>thought\nConsider x.')
        await hold.wait
        yield part('<channel|>After', 'stop')
      })(),
    )
    const stream = model.stream(messages, { toolSpecs: tools })
    try {
      const events = []
      for (;;) {
        const event = (await stream.next()).value!
        events.push(event)
        if (
          event.type === 'modelContentBlockDeltaEvent' &&
          event.delta.type === 'reasoningContentDelta'
        )
          break
      }
      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'textDelta', text: 'Before' },
      })
      expect(events.at(-1)).toEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'reasoningContentDelta', text: 'Consider x.' },
      })
      hold.resume()
      const remaining = await collect(stream)
      expect(remaining.slice(0, 3)).toEqual([
        { type: 'modelContentBlockStopEvent' },
        { type: 'modelContentBlockStartEvent' },
        { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'After' } },
      ])
    } finally {
      hold.resume()
      await stream.return(undefined)
    }
  })

  it('treats control markers inside tool arguments as literal data', async () => {
    const { model } = setup(
      ...Array.from(
        '<|tool_call>call:lookup{key:<|"|><|channel>thought\ntext<channel|><|"|>}<tool_call|>',
        (text) => part(text),
      ),
      part('', 'stop'),
    )
    const result = await aggregate(model)
    expect(result.message.content).toEqual([
      expect.objectContaining({
        type: 'toolUseBlock',
        input: { key: '<|channel>thought\ntext<channel|>' },
      }),
    ])
  })

  it.each(['<|channel>analysis\nx<channel|>', '<|channel>thought', '<|channel>thought\nunclosed'])(
    'rejects malformed thinking without executing tools: %s',
    async (source) => {
      const { model } = setup(part(source, 'stop'))
      const callback = vi.fn()
      const agent = new Agent({
        model,
        printer: false,
        tools: [new FunctionTool({ ...tools[0]!, callback })],
      })
      await expect(agent.invoke('Find x.')).rejects.toThrow(/thinking/)
      expect(callback).not.toHaveBeenCalled()
    },
  )

  it('does not execute tool syntax inside truncated thinking', async () => {
    const { model } = setup(part('<|channel>thought\n' + call, 'length'))
    const callback = vi.fn()
    const agent = new Agent({
      model,
      printer: false,
      tools: [new FunctionTool({ ...tools[0]!, callback })],
    })
    await expect(agent.invoke('Find x.')).rejects.toBeInstanceOf(MaxTokensError)
    expect(callback).not.toHaveBeenCalled()
  })

  it('supports thinking without tools and can turn it off with reasoning still in Agent history', async () => {
    const { model, create } = setup(part('<|channel>thought\nCheck.<channel|>42', 'stop'))
    model.updateConfig({ request: { extra_body: { enable_thinking: true } } })
    const agent = new Agent({ model, printer: false })
    expect((await agent.invoke('Answer.')).lastMessage.content).toEqual([
      new ReasoningBlock({ text: 'Check.' }),
      new TextBlock('42'),
    ])
    model.updateConfig({ request: {} })
    create.mockResolvedValueOnce(parts(part('Still 42', 'stop')))
    expect(String(await agent.invoke('Again.'))).toBe('Still 42')
    expect(create.mock.calls[1]![0].prompt).not.toContain('<|think|>')
    expect(create.mock.calls[1]![0].prompt).not.toContain('Check.')
  })

  it('can cancel a reasoning delta, then reuse the engine without leaking parser state', async () => {
    const { model, create, stop } = setup(part('<|channel>thought\nPartial'), part('', 'stop'))
    model.updateConfig({ request: { extra_body: { enable_thinking: true } } })
    const controller = new AbortController()
    const agent = new Agent({ model, printer: false })
    for await (const update of agent.stream('Think.', { cancelSignal: controller.signal })) {
      if (
        update.type === 'modelStreamUpdateEvent' &&
        update.event.type === 'modelContentBlockDeltaEvent' &&
        update.event.delta.type === 'reasoningContentDelta'
      )
        controller.abort()
    }
    expect(controller.signal.aborted).toBe(true)
    expect(stop).toHaveBeenCalledTimes(1)
    create.mockResolvedValueOnce(parts(part('<|channel>thought\nNew.<channel|>Recovered', 'stop')))
    expect((await aggregate(model)).message.content).toEqual([
      new ReasoningBlock({ text: 'New.' }),
      new TextBlock('Recovered'),
    ])
  })
  it('runs an actual Strands tool loop through raw completions and returns the model answer', async () => {
    const { model, create, chat } = setup()
    const lookup = vi.fn<(input: unknown) => string>(
      () => 'Only the tool knows this result: secret-42',
    )
    create.mockResolvedValueOnce(parts(...Array.from(call, (text) => part(text)), part('', 'stop')))
    create.mockResolvedValueOnce(parts(part('The result is secret-42.', 'stop')))
    const agent = new Agent({
      model,
      printer: false,
      tools: [new FunctionTool({ ...tools[0]!, callback: lookup })],
    })
    const result = await agent.invoke('Find x.')
    expect(lookup.mock.calls[0]?.[0]).toEqual({ key: 'x' })
    expect(String(result)).toBe('The result is secret-42.')
    expect(create).toHaveBeenCalledTimes(2)
    expect(chat).not.toHaveBeenCalled()
    const request = create.mock.calls[1]![0]
    expect(request.prompt).toContain(
      '<|tool_response>response:lookup{value:<|"|>Only the tool knows this result: secret-42<|"|>}<tool_response|>',
    )
    expect(request).not.toHaveProperty('tools')
    expect(request).not.toHaveProperty('messages')
  })

  it('streams normal text while holding split tool markers and never exposes tool syntax', async () => {
    const { model, create } = setup()
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield part('Let me check. <|too')
        await hold.wait
        yield part('l_call>call:lookup{}<tool_call|>')
        yield part('', 'stop')
      })(),
    )
    const stream = model.stream(messages, { toolSpecs: tools })
    try {
      await stream.next()
      await stream.next()
      expect((await stream.next()).value).toMatchObject({ delta: { text: 'Let me check. ' } })
      hold.resume()
      const remaining = await collect(stream)
      expect(remaining).toContainEqual(
        expect.objectContaining({ start: expect.objectContaining({ name: 'lookup' }) }),
      )
      expect(JSON.stringify(remaining)).not.toContain('<|tool_call>')
    } finally {
      hold.resume()
      await stream.return(undefined)
    }
  })

  it('preserves ordinary text that only partly resembles a marker', async () => {
    const { model } = setup(part('Use <'), part('value>.'), part('<', 'stop'))
    const result = await aggregate(model)
    expect(result.message.content).toEqual([new TextBlock('Use <value>.<')])
  })

  it('keeps multiple calls and nested argument values separate', async () => {
    const source =
      '<|tool_call>call:lookup{key:<|"|>a<|"|>,options:{list:[1,true,null]}}<tool_call|>\n' + call
    const { model } = setup(part(source, 'stop'))
    const result = await aggregate(model)
    const calls = result.message.content.filter((block) => block.type === 'toolUseBlock')
    expect(calls.map((block) => block.input)).toEqual([
      { key: 'a', options: { list: [1, true, null] } },
      { key: 'x' },
    ])
    expect(new Set(calls.map((block) => block.toolUseId)).size).toBe(2)
  })

  it.each([
    '<|tool_call>call:unknown{}<tool_call|>',
    '<|tool_call>broken',
    '<|tool_call>call:lookup{key:undefined}<tool_call|>',
    '<|tool_call>call:lookup{}',
    call + 'unexpected text',
  ])('rejects malformed calls without executing tools: %s', async (source) => {
    const { model } = setup(part(source, 'stop'))
    const callback = vi.fn()
    const agent = new Agent({
      model,
      printer: false,
      tools: [new FunctionTool({ ...tools[0]!, callback })],
    })
    await expect(agent.invoke('Find x.')).rejects.toThrow()
    expect(callback).not.toHaveBeenCalled()
  })

  it('never executes a tool when generation is truncated, even after a complete call', async () => {
    const { model } = setup(part(call, 'length'))
    const callback = vi.fn()
    const agent = new Agent({
      model,
      printer: false,
      tools: [new FunctionTool({ ...tools[0]!, callback })],
    })
    await expect(agent.invoke('Find x.')).rejects.toBeInstanceOf(MaxTokensError)
    expect(callback).not.toHaveBeenCalled()
  })

  it('never executes a tool when the native engine aborts', async () => {
    const { model } = setup(part(call, 'abort'))
    const result = await aggregate(model)
    expect(result.stopReason).toBe('cancelled')
    expect(result.message.content).toEqual([])
  })

  it('passes sampling settings, usage, probabilities and original native chunks through', async () => {
    const { engine, create } = setup()
    const onChunk = vi.fn()
    const value = part('Done', 'stop')
    value.choices[0]!.logprobs = { content: [] }
    const usage = {
      ...part(),
      choices: [],
      usage: {
        prompt_tokens: 5,
        completion_tokens: 2,
        total_tokens: 7,
        extra: {
          e2e_latency_s: 1,
          prefill_tokens_per_s: 10,
          decode_tokens_per_s: 2,
          time_to_first_token_s: 0.1,
          time_per_output_token_s: 0.5,
        },
      },
    }
    create.mockResolvedValueOnce(parts(value, usage))
    const model = new WebLLMModel({
      engine,
      modelId,
      maxTokens: 64,
      temperature: 0.2,
      topP: 0.8,
      onChunk,
      request: { seed: 7, extra_body: { enable_latency_breakdown: true } },
    })
    const result = await aggregate(model)
    expect(result.metadata?.usage).toEqual({ inputTokens: 5, outputTokens: 2, totalTokens: 7 })
    expect(onChunk.mock.calls.map(([chunk]) => chunk)).toEqual([value, usage])
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        seed: 7,
        max_tokens: 64,
        temperature: 0.2,
        top_p: 0.8,
        stream_options: { include_usage: true },
        n: 1,
        extra_body: { enable_latency_breakdown: true },
      }),
    )
  })

  it('can select a named registered tool without asking the model to invent the prefix', async () => {
    const { model, create } = setup(part('}<tool_call|>', 'stop'))
    const result = await collect(
      model.stream(messages, { toolSpecs: tools, toolChoice: { tool: { name: 'lookup' } } }),
    )
    expect(create.mock.calls[0]?.[0].prompt.endsWith('<|tool_call>call:lookup{')).toBe(true)
    expect(result).toContainEqual(
      expect.objectContaining({ start: expect.objectContaining({ name: 'lookup' }) }),
    )
    await expect(
      collect(
        model.stream(messages, { toolSpecs: tools, toolChoice: { tool: { name: 'missing' } } }),
      ),
    ).rejects.toThrow('not registered')
    await expect(
      collect(model.stream(messages, { toolSpecs: tools, toolChoice: { any: {} } })),
    ).rejects.toThrow('toolChoice.any')
  })

  it('starts a named tool at its first required field while the model supplies the value', async () => {
    const { model, create } = setup(part('/tmp/note.txt"}<tool_call|>', 'stop'))
    const tool: ToolSpec = {
      name: 'read',
      description: 'Read a file.',
      inputSchema: {
        type: 'object',
        properties: { path: { type: 'string' } },
        required: ['path'],
      },
    }
    const events = await collect(
      model.stream(messages, { toolSpecs: [tool], toolChoice: { tool: { name: 'read' } } }),
    )
    expect(create.mock.calls[0]?.[0].prompt.endsWith('<|tool_call>call:read{path:"')).toBe(true)
    expect(events).toContainEqual(
      expect.objectContaining({
        delta: { type: 'toolUseInputDelta', input: '{"path":"/tmp/note.txt"}' },
      }),
    )
  })

  it('can generate an empty required array after a named tool prefix', async () => {
    const { model, create } = setup(part(']}<tool_call|>', 'stop'))
    const tool: ToolSpec = {
      name: 'clear',
      description: 'Clear the list.',
      inputSchema: {
        type: 'object',
        properties: { todos: { type: 'array', items: { type: 'string' } } },
        required: ['todos'],
      },
    }
    const events = await collect(
      model.stream(messages, { toolSpecs: [tool], toolChoice: { tool: { name: 'clear' } } }),
    )
    expect(create.mock.calls[0]?.[0].prompt.endsWith('<|tool_call>call:clear{todos:[')).toBe(true)
    expect(events).toContainEqual(
      expect.objectContaining({ delta: { type: 'toolUseInputDelta', input: '{"todos":[]}' } }),
    )
  })

  it('allows JSON-escaped control markers in a named string argument', async () => {
    const value = '<|"|>literal<|channel>thought\n'
    const { model } = setup(part(JSON.stringify(value).slice(1) + '}<tool_call|>', 'stop'))
    const tool: ToolSpec = {
      name: 'read',
      description: 'Read a path.',
      inputSchema: {
        type: 'object',
        properties: { path: { type: 'string' } },
        required: ['path'],
      },
    }
    const events = await collect(
      model.stream(messages, { toolSpecs: [tool], toolChoice: { tool: { name: 'read' } } }),
    )
    expect(events).toContainEqual(
      expect.objectContaining({
        delta: { type: 'toolUseInputDelta', input: JSON.stringify({ path: value }) },
      }),
    )
  })

  it.each([
    { response_format: { type: 'json_object' as const } },
    { extra_body: { resumable: { enabled: true } } },
    { ignore_eos: true },
  ])('rejects incompatible options before opening a native call', async (request) => {
    const { model, create } = setup()
    model.updateConfig({ request })
    await expect(aggregate(model)).rejects.toThrow('Gemma protocol mode does not support')
    expect(create).not.toHaveBeenCalled()
  })

  it('releases a cancelled Gemma call even if Strands abandons the model iterator', async () => {
    const { model, engine, create, stop } = setup(part('Partial'), part('', 'stop'))
    const controller = new AbortController()
    const agent = new Agent({
      model,
      tools: [new FunctionTool({ ...tools[0]!, callback: () => 'ok' })],
      printer: false,
    })
    for await (const update of agent.stream('Find x.', { cancelSignal: controller.signal })) {
      if (
        update.type === 'modelStreamUpdateEvent' &&
        update.event.type === 'modelContentBlockDeltaEvent'
      )
        controller.abort()
    }
    expect(stop).toHaveBeenCalledTimes(1)
    create.mockResolvedValueOnce(parts(part('Recovered', 'stop')))
    expect((await aggregate(model)).message.content).toEqual([new TextBlock('Recovered')])
    // Both APIs share the same engine ownership guard.
    const chat = new WebLLMModel({ engine, modelId: 'other' })
    const native = vi.spyOn(engine.chat.completions, 'create').mockResolvedValue(
      (async function* () {
        yield chunk({ content: 'Chat' }, 'stop')
      })(),
    )
    await collect(chat.stream(messages))
    expect(native).toHaveBeenCalledTimes(1)
  })

  it('cleans up after callback errors and rejects extra completion choices', async () => {
    const { model, create, engine } = setup()
    const broken = new WebLLMModel({
      engine,
      modelId,
      onChunk: () => {
        throw new Error('Callback failed')
      },
    })
    create.mockResolvedValueOnce(parts(part('Hi', 'stop')))
    await expect(aggregate(broken)).rejects.toThrow('Callback failed')
    const value = part('Hi', 'stop')
    value.choices[0]!.index = 1
    create.mockResolvedValueOnce(parts(value))
    await expect(aggregate(model)).rejects.toThrow('one response')
    create.mockResolvedValueOnce(parts(part('Recovered', 'stop')))
    expect((await aggregate(model)).stopReason).toBe('endTurn')
  })

  it('keeps Gemma chat without tools on the native chat API', async () => {
    const { engine, create } = chatSetup(chunk({ content: 'Hi' }, 'stop'))
    const model = new WebLLMModel({ engine, modelId })
    await collect(model.stream(messages))
    expect(create).toHaveBeenCalledTimes(1)
  })
})
