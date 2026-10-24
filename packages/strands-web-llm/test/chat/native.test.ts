import { setImmediate as nextTurn } from 'node:timers/promises'
import { describe, expect, it, vi } from 'vitest'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '@strands-agents/sdk'
import { WebLLMModel, withEngine } from '../../src/index.js'
import { chunk, collect, pause, setup } from '../../fixtures/engine.js'

const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]
const tools = [{ name: 'weather', description: 'Get the weather' }]

describe('native WebLLM messages', () => {
  it('does not observe late native chunks after cancellation', async () => {
    const { engine, create, stop } = setup()
    const waiting = pause()
    const hold = pause()
    const first = chunk({ content: 'Hi' })
    create.mockResolvedValueOnce(
      (async function* () {
        yield first
        waiting.resume()
        await hold.wait
        yield chunk({ content: 'Late' }, 'abort')
      })(),
    )
    stop.mockImplementation(async () => hold.resume())
    const onChunk = vi.fn()
    const model = new WebLLMModel({ engine, modelId: 'test-model', onChunk })
    const controller = new AbortController()
    const pending = collect(model.stream(messages, { cancelSignal: controller.signal }))
    await waiting.wait
    controller.abort()

    expect(await pending).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'cancelled' })
    expect(onChunk).toHaveBeenCalledExactlyOnceWith(first)
  })

  it.each(['error', 'end'])('keeps pending JSON before a native stream %s', async (finish) => {
    const raw = '[{"name":"weather","arguments":{'
    const failure = new Error('Native stream failed')
    const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: raw.slice(0, 12) })
        yield chunk({ content: raw.slice(12) })
        if (finish === 'error') throw failure
      })(),
    )
    const output: string[] = []
    const pending = (async () => {
      for await (const event of model.stream(messages, { toolSpecs: tools })) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          output.push(event.delta.text)
        if (event.type === 'modelContentBlockStartEvent') expect(event.start).toBeUndefined()
        expect(event.type).not.toBe('modelMessageStopEvent')
      }
    })()

    if (finish === 'error') await expect(pending).rejects.toBe(failure)
    else await expect(pending).rejects.toThrow('without a stop reason')
    expect(output.join('')).toBe(raw)
    expect(await collect(model.stream(messages))).toContainEqual({
      type: 'modelMessageStopEvent',
      stopReason: 'endTurn',
    })
  })

  it('keeps Hermes tool arguments in the content WebLLM reads', async () => {
    const { engine, create } = setup(chunk({ content: 'Seoul' }, 'stop'))
    const model = new WebLLMModel({ engine, modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' })
    const history = [
      ...messages,
      new Message({
        role: 'assistant',
        content: [
          new ToolUseBlock({ name: 'weather', toolUseId: 'call-1', input: { city: 'Seoul' } }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'call-1',
            status: 'success',
            content: [new TextBlock('Done.')],
          }),
        ],
      }),
    ]
    await collect(model.stream(history))
    expect(create.mock.calls[0]?.[0].messages[1]).toMatchObject({
      role: 'assistant',
      content: '[{"name":"weather","arguments":{"city":"Seoul"}}]',
    })
    expect(history[1]?.content).toHaveLength(1)
    expect(history[1]?.content[0]?.type).toBe('toolUseBlock')
  })

  it('snapshots tool schemas and choices before waiting for the engine', async () => {
    const { engine, create } = setup(
      chunk(
        {
          tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{"city":"Seoul"}' } }],
        },
        'tool_calls',
      ),
    )
    const model = new WebLLMModel({ engine, modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' })
    const hold = pause()
    const active = withEngine(engine, () => hold.wait)
    const toolSpecs = [
      {
        name: 'weather',
        description: 'Get the weather',
        inputSchema: {
          type: 'object' as const,
          properties: { city: { type: 'string' as const, description: 'Original schema' } },
        },
      },
    ]
    const toolChoice = { tool: { name: 'weather' } }
    const pending = collect(model.stream(messages, { toolSpecs, toolChoice }))
    try {
      await nextTurn()
      toolSpecs[0]!.name = 'changed'
      toolSpecs[0]!.inputSchema.properties.city = { type: 'string', description: 'Changed schema' }
      toolChoice.tool.name = 'changed'
      hold.resume()
      await active
      const events = await pending
      expect(create.mock.calls[0]?.[0].tools?.[0]?.function).toMatchObject({
        name: 'weather',
        parameters: { properties: { city: { type: 'string', description: 'Original schema' } } },
      })
      expect(events.at(-1)).toEqual({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
    } finally {
      hold.resume()
      await Promise.allSettled([active, pending])
    }
  })

  it('streams native commentary before generation finishes with tools registered', async () => {
    const { model, create } = setup()
    const hold = pause()
    create.mockImplementationOnce(async () =>
      (async function* () {
        yield chunk({ content: 'Checking now.' })
        await hold.wait
        yield chunk({}, 'stop')
      })(),
    )
    const stream = model.stream(messages, { toolSpecs: tools })
    try {
      await stream.next()
      await stream.next()
      expect(await stream.next()).toMatchObject({
        value: {
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: 'Checking now.' },
        },
      })
      hold.resume()
      await collect(stream)
    } finally {
      hold.resume()
      await stream.return(undefined)
    }
  })

  it.each([
    'Checking the weather now.',
    '["Seoul"]',
    '[1e999]',
    '[{"name":"weather","arguments":{"city":"Seoul"},"note":"Keep this note."}]',
  ])('preserves text arriving before a native tool call: %s', async (text) => {
    const { model } = setup(
      ...text.split('').map((content) => chunk({ content })),
      chunk(
        {
          tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{"city":"Seoul"}' } }],
        },
        'tool_calls',
      ),
    )
    const events = await collect(model.stream(messages, { toolSpecs: tools }))
    const output = events
      .flatMap((event) =>
        event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta'
          ? [event.delta.text]
          : [],
      )
      .join('')
    expect(output).toBe(text)
    expect(events.at(-1)).toEqual({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
  })

  it('keeps assistant commentary beside a native tool call', async () => {
    const { model } = setup(
      chunk(
        {
          content: 'Checking the weather now.',
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              function: { name: 'weather', arguments: '{"city":"Seoul"}' },
            },
          ],
        },
        'tool_calls',
      ),
    )

    const events = await collect(model.stream(messages, { toolSpecs: tools }))
    expect(events).toContainEqual({
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'textDelta', text: 'Checking the weather now.' },
    })
    expect(events).toContainEqual({
      type: 'modelContentBlockStartEvent',
      start: { type: 'toolUseStart', name: 'weather', toolUseId: 'call-1' },
    })
    expect(events).toContainEqual({
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input: '{"city":"Seoul"}' },
    })
  })

  it.each(['1e999', '-1e999'])(
    'rejects native arguments changed to null by overflow: %s',
    async (number) => {
      const raw = `[{"name":"weather","arguments":{"city":"Seoul","temperature":${number}}}]`
      const args = JSON.stringify(JSON.parse(raw)[0].arguments)
      const { model } = setup(
        chunk({ content: raw }),
        chunk(
          { tool_calls: [{ index: 0, function: { name: 'weather', arguments: args } }] },
          'tool_calls',
        ),
      )

      await expect(collect(model.stream(messages, { toolSpecs: tools }))).rejects.toThrow(
        'finite numbers',
      )
    },
  )

  it.each([
    '{"city":"Seoul","unit":"C"}',
    '{ "city": "Seoul", "unit": "C" }',
    '{"unit":"C","city":"Seoul"}',
  ])('suppresses duplicate tool JSON while keeping native arguments: %s', async (input) => {
    const { model } = setup(
      chunk(
        {
          content: '[{"name":"weather","arguments":{"city":"Seoul","unit":"C"}}]',
          tool_calls: [
            {
              index: 0,
              id: 'call-1',
              function: { name: 'weather', arguments: input },
            },
          ],
        },
        'tool_calls',
      ),
    )

    const events = await collect(model.stream(messages, { toolSpecs: tools }))
    expect(
      events.some(
        (event) => event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta',
      ),
    ).toBe(false)
    expect(events).toContainEqual({
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input },
    })
    expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
  })
})
