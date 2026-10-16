import { describe, expect, it, vi } from 'vitest'
import {
  Agent,
  FunctionTool,
  MaxTokensError,
  Message,
  TextBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import type { ModelStreamEvent } from '@strands-agents/sdk'
import { WebLLMModel } from '../src/index.js'
import { chunk, chunks, collect, setup } from './helpers.js'

const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]
const tools = [{ name: 'weather', description: 'Get the weather' }]

async function aggregate(model: WebLLMModel) {
  const stream = model.streamAggregated(messages)
  for (;;) {
    const step = await stream.next()
    if (step.done) return step.value
  }
}

describe('WebLLMModel', () => {
  it('streams text through a real Strands Agent', async () => {
    const { model, create } = setup(chunk({ content: 'Hello' }), chunk({}, 'stop'))
    const agent = new Agent({ model, printer: false })
    const result = await agent.invoke('Hi')
    expect(result.stopReason).toBe('endTurn')
    expect(result.lastMessage?.content).toEqual([new TextBlock('Hello')])
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'test-model',
        stream: true,
        stream_options: { include_usage: true },
        n: 1,
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    )
  })

  it('emits text before waiting for the rest of the response', async () => {
    const { model } = setup(chunk({ content: 'Hi' }), chunk({}, 'stop'))
    const stream = model.stream(messages)
    expect((await stream.next()).value).toMatchObject({ type: 'modelMessageStartEvent' })
    expect((await stream.next()).value).toMatchObject({ type: 'modelContentBlockStartEvent' })
    expect((await stream.next()).value).toMatchObject({ delta: { text: 'Hi' } })
    await stream.return(undefined)
  })

  it('keeps native request options and raw chunks available', async () => {
    const part = chunk({ content: 'Hi' }, 'stop')
    const onChunk = vi.fn()
    const { engine, create } = setup(part)
    const model = new WebLLMModel({
      engine,
      modelId: 'test-model',
      maxTokens: 64,
      temperature: 0,
      topP: 0.9,
      onChunk,
      request: { seed: 7, logprobs: true, repetition_penalty: 1.1 },
    })
    await collect(model.stream(messages))
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        seed: 7,
        logprobs: true,
        repetition_penalty: 1.1,
        max_tokens: 64,
        temperature: 0,
        top_p: 0.9,
      }),
    )
    expect(onChunk).toHaveBeenCalledWith(part)
    expect(model.engine).toBe(engine)
  })

  it('copies config so WebLLM cannot mutate the next request', () => {
    const { engine } = setup()
    const request = { temperature: 0.5 }
    const model = new WebLLMModel({ engine, modelId: 'first', request })
    request.temperature = 1
    const copy = model.getConfig()
    copy.request!.temperature = 2
    expect(model.getConfig().request?.temperature).toBe(0.5)
    model.updateConfig({ modelId: 'second', maxTokens: 10 })
    expect(model.getConfig()).toMatchObject({ modelId: 'second', maxTokens: 10 })
    model.updateConfig({ temperature: 0.2 })
    expect(model.getConfig().temperature).toBe(0.2)
  })

  it('keeps the final usage chunk after the stop choice', async () => {
    const usage = {
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
    }
    const { model } = setup(chunk({ content: 'Hi' }, 'stop'), {
      ...chunk(),
      choices: [],
      usage,
    })
    const result = await aggregate(model)
    expect(result.metadata?.usage).toEqual({ inputTokens: 5, outputTokens: 2, totalTokens: 7 })
    expect(result.metadata?.trace).toEqual(usage.extra)
  })

  it('runs a tool and sends its result back through the Agent loop', async () => {
    const { model, create } = setup()
    create.mockResolvedValueOnce(
      chunks(
        chunk({ content: '[{"name":"weather","arguments":{}}]' }),
        chunk(
          {
            tool_calls: [
              { index: 0, id: 'call-1', function: { name: 'weather', arguments: '{}' } },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Sunny' }, 'stop')))
    const weather = new FunctionTool({ ...tools[0]!, callback: () => 'Sunny' })
    const agent = new Agent({ model, tools: [weather], printer: false })
    const result = await agent.invoke('Weather?')
    expect(result.lastMessage?.content).toEqual([new TextBlock('Sunny')])
    expect(create).toHaveBeenCalledTimes(2)
    const request = create.mock.calls[1]![0]
    expect(request.messages).toContainEqual({
      role: 'tool',
      tool_call_id: 'call-1',
      content: 'Sunny',
    })
    const assistant = request.messages.find((message) => message.role === 'assistant')
    expect(assistant).toMatchObject({ content: null, tool_calls: [{ id: 'call-1' }] })
  })

  it('joins split tool names and JSON without showing raw tool JSON as text', async () => {
    const { model } = setup(
      chunk({
        content: 'raw tool JSON',
        tool_calls: [{ index: 0, id: 'a', function: { name: 'wea', arguments: '{"city":' } }],
      }),
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'ther', arguments: '"Seoul"}' } }] },
        'tool_calls',
      ),
    )
    const stream = model.streamAggregated(messages, { toolSpecs: tools })
    const blocks = []
    for await (const event of stream) {
      if (event instanceof TextBlock || event instanceof ToolUseBlock) blocks.push(event)
    }
    expect(blocks).toEqual([
      new ToolUseBlock({ name: 'weather', toolUseId: 'a', input: { city: 'Seoul' } }),
    ])
  })

  it('never turns a truncated tool call into executable tool use', async () => {
    const { model } = setup(
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{' } }] },
        'length',
      ),
    )
    await expect(aggregate(model)).rejects.toBeInstanceOf(MaxTokensError)
  })

  it('rejects broken tool JSON', async () => {
    const { model } = setup(
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{' } }] },
        'tool_calls',
      ),
    )
    await expect(aggregate(model)).rejects.toThrow('invalid tool JSON')
  })

  it('rejects a stream without a stop reason', async () => {
    const { model } = setup(chunk({ content: 'cut off' }))
    await expect(aggregate(model)).rejects.toThrow('without a stop reason')
  })

  it('does not invent tools when the backend returns none', async () => {
    const { model } = setup(chunk({}, 'tool_calls'))
    await expect(aggregate(model)).rejects.toThrow('no tool calls')
  })

  it('rejects a tool choice WebLLM cannot represent', async () => {
    const { model, create } = setup()
    await expect(collect(model.stream(messages, { toolChoice: { any: {} } }))).rejects.toThrow(
      'toolChoice.any',
    )
    expect(create).not.toHaveBeenCalled()
  })

  it('does not start an already cancelled call', async () => {
    const { model, create, stop } = setup()
    await expect(
      collect(model.stream(messages, { cancelSignal: AbortSignal.abort() })),
    ).rejects.toThrow()
    expect(create).not.toHaveBeenCalled()
    expect(stop).not.toHaveBeenCalled()
  })

  it('interrupts generation when cancelled and removes the listener', async () => {
    const { model, stop } = setup(chunk({ content: 'Hi' }), chunk({}, 'abort'))
    const controller = new AbortController()
    const stream = model.stream(messages, { cancelSignal: controller.signal })
    await stream.next()
    controller.abort()
    expect(stop).toHaveBeenCalledTimes(1)
    const events = await collect(stream)
    expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
    expect(events.some((event) => event.type === 'modelContentBlockDeltaEvent')).toBe(false)
  })

  it('stops and releases the engine when a consumer leaves early', async () => {
    const { model, stop, create } = setup(chunk({ content: 'Hi' }), chunk({}, 'stop'))
    const stream = model.stream(messages)
    await stream.next()
    await stream.return(undefined)
    expect(stop).toHaveBeenCalledTimes(1)
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Again' }, 'stop')))
    await expect(aggregate(model)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('rejects overlapping calls even when two adapters share an engine', async () => {
    const { model, engine, stop } = setup(chunk({ content: 'Hi' }, 'stop'))
    const first = model.stream(messages)
    await first.next()
    const second = new WebLLMModel({ engine, modelId: 'other' })
    await expect(aggregate(second)).rejects.toThrow('already in use')
    expect(stop).not.toHaveBeenCalled()
    await first.return(undefined)
  })

  it('releases the engine after a backend error', async () => {
    const { model, create } = setup()
    create.mockRejectedValueOnce(new Error('GPU lost'))
    await expect(aggregate(model)).rejects.toThrow('GPU lost')
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Recovered' }, 'stop')))
    await expect(aggregate(model)).resolves.toMatchObject({ stopReason: 'endTurn' })
  })

  it('keeps the event order valid for an empty response', async () => {
    const { model } = setup(chunk({}, 'stop'))
    const events: ModelStreamEvent[] = await collect(model.stream(messages))
    expect(events).toEqual([
      { type: 'modelMessageStartEvent', role: 'assistant' },
      { type: 'modelMessageStopEvent', stopReason: 'endTurn' },
    ])
  })
})
