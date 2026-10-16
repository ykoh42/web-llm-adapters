import { describe, expect, it, vi } from 'vitest'
import { Agent, FunctionTool, MaxTokensError, Message, TextBlock } from '@strands-agents/sdk'
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
    const { model, create } = setup(part('{}<tool_call|>', 'stop'))
    const result = await collect(
      model.stream(messages, { toolSpecs: tools, toolChoice: { tool: { name: 'lookup' } } }),
    )
    expect(create.mock.calls[0]?.[0].prompt.endsWith('<|tool_call>call:lookup')).toBe(true)
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

  it.each([
    { response_format: { type: 'json_object' as const } },
    { extra_body: { enable_thinking: true } },
    { extra_body: { resumable: { enabled: true } } },
    { ignore_eos: true },
  ])('rejects incompatible options before opening a native call', async (request) => {
    const { model, create } = setup()
    model.updateConfig({ request })
    await expect(aggregate(model)).rejects.toThrow('Gemma tool mode does not support')
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
