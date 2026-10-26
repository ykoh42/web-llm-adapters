import { describe, expect, it, vi } from 'vitest'
import { AIMessage, AIMessageChunk, HumanMessage, ToolMessage } from '@langchain/core/messages'
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event'
import { ChatWebLLM } from '../src/index.js'
import { readChunks } from '../src/chat/chunks.js'
import { chunk, collect, pause, setup } from '../fixtures/engine.js'
import { nativeChunks } from '../../../shared/web-llm/native.js'

const lookup = {
  type: 'function',
  function: {
    name: 'lookup',
    description: 'Look up a code.',
    parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] },
  },
}

describe('native WebLLM messages', () => {
  it('keeps reasoning, tool calls, and later text in separate event blocks', async () => {
    const { engine } = setup(
      chunk({ content: '<think>Thought</think>' }),
      chunk({ content: '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>' }),
      chunk({ content: 'Do' }),
      chunk({ content: 'ne' }, 'stop'),
    )
    const model = new ChatWebLLM({ engine, modelId: 'Qwen3-4B-q4f16_1-MLC' })
    const stream = model.streamEvents('Look up A7.', { tools: [lookup] })
    const events = await collect(stream)

    expect(events.filter((event) => event.event === 'content-block-finish')).toEqual([
      expect.objectContaining({ index: 0, content: { type: 'reasoning', reasoning: 'Thought' } }),
      expect.objectContaining({
        index: 1,
        content: expect.objectContaining({
          type: 'tool_call',
          name: 'lookup',
          args: { code: 'A7' },
        }),
      }),
      expect.objectContaining({ index: 2, content: { type: 'text', text: 'Done' } }),
    ])
    expect((await stream).text).toBe('Done')
    expect((await stream.toolCalls).map(({ name, args }) => ({ name, args }))).toEqual([
      { name: 'lookup', args: { code: 'A7' } },
    ])
  })

  it.each(['error', 'end'])('keeps pending JSON before a native stream %s', async (finish) => {
    const raw = '[{"name":"lookup","arguments":{'
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
      for await (const item of await model.bindTools([lookup]).stream('Look up A7.')) {
        expect(item.tool_calls ?? []).toEqual([])
        output.push(item.text)
      }
    })()

    if (finish === 'error') await expect(pending).rejects.toBe(failure)
    else await expect(pending).rejects.toThrow('without a stop reason')
    expect(output.join('')).toBe(raw)
    expect((await model.invoke('Try again.')).text).toBe('Recovered')
  })

  it.each(['Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC', 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'])(
    'replays native tool arguments as assistant content for %s',
    async (modelId) => {
      const { engine, create } = setup(chunk({ content: 'A7' }, 'stop'))
      const model = new ChatWebLLM({ engine, modelId })
      const history = [
        new HumanMessage('Look up the code.'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call-1', name: 'lookup', args: { code: 'A7' } }],
        }),
        new ToolMessage({ tool_call_id: 'call-1', content: 'Done.' }),
        new HumanMessage('Which code?'),
      ]
      await model.invoke(history)
      expect(create.mock.calls[0]?.[0].messages[1]).toMatchObject({
        role: 'assistant',
        content: '[{"name":"lookup","arguments":{"code":"A7"}}]',
      })
      expect(history[1]?.content).toBe('')
    },
  )

  it.each([
    ' [ { "name": "lookup", "arguments": { "code": "A7" } } ] ',
    '[{"name":"lookup","arguments":{"code":"A7"},"note":"Keep this note."}]',
  ])('keeps native tool text without duplicating it in history: %s', async (raw) => {
    const { engine, create } = setup(chunk({ content: 'A7' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' })
    await model.invoke([
      new HumanMessage('Look up the code.'),
      new AIMessage({
        content: raw,
        tool_calls: [{ id: 'call-1', name: 'lookup', args: { code: 'A7' } }],
      }),
      new ToolMessage({ tool_call_id: 'call-1', content: 'Done.' }),
    ])
    expect(create.mock.calls[0]?.[0].messages[1]).toMatchObject({ content: raw })
  })

  it('keeps native tool order when argument chunks arrive out of order', async () => {
    const { model } = setup(
      chunk({
        tool_calls: [
          { index: 1, id: 'second', function: { name: 'lookup', arguments: '{"code":' } },
        ],
      }),
      chunk({
        tool_calls: [
          { index: 0, id: 'first', function: { name: 'lookup', arguments: '{"code":"A7"}' } },
        ],
      }),
      chunk({ tool_calls: [{ index: 1, function: { arguments: '"A8"}' } }] }, 'tool_calls'),
    )
    const bound = model.bindTools([lookup])
    expect(
      (await bound.invoke('Look up both codes.')).tool_calls?.map((call) => call.args),
    ).toEqual([{ code: 'A7' }, { code: 'A8' }])
    const events = model.streamEvents('Look up both codes.', { tools: [lookup] })
    expect((await events.toolCalls).map((call) => call.args)).toEqual([
      { code: 'A7' },
      { code: 'A8' },
    ])
  })

  it('accepts a tool call ID after an empty first delta', async () => {
    const { model } = setup(
      chunk({
        tool_calls: [{ index: 0, id: '', function: { name: 'look', arguments: '{"code":' } }],
      }),
      chunk(
        {
          tool_calls: [{ index: 0, id: 'call-1', function: { name: 'up', arguments: '"A7"}' } }],
        },
        'tool_calls',
      ),
    )

    const result = await model.bindTools([lookup]).invoke('Look up A7.')

    expect(result.tool_calls).toMatchObject([
      { id: 'call-1', name: 'lookup', args: { code: 'A7' } },
    ])
  })

  it('streams native tool-call chunks before generation finishes', async () => {
    const { model, create } = setup()
    const yielded = pause()
    const resume = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yielded.resume()
        yield chunk({
          tool_calls: [
            { index: 0, id: 'call-1', function: { name: 'look', arguments: '{"code":' } },
          ],
        })
        await resume.wait
        yield chunk({
          tool_calls: [{ index: 0, function: { name: 'up', arguments: '"A7"}' } }],
        })
        yield chunk({}, 'tool_calls')
      })(),
    )

    const pending = model.bindTools([lookup]).stream('Look up A7.')
    await yielded.wait
    const first = await Promise.race([
      pending.then((stream) => ({ stream })),
      new Promise<{ timeout: true }>((resolve) => setTimeout(() => resolve({ timeout: true }), 0)),
    ])
    resume.resume()

    const stream = 'stream' in first ? first.stream : await pending
    const messages: AIMessageChunk[] = []
    for await (const item of stream) messages.push(item)

    expect('timeout' in first).toBe(false)
    expect(messages[0]?.tool_call_chunks).toEqual([
      expect.objectContaining({ index: 0, name: 'look', args: '{"code":' }),
    ])
    expect(messages[0]?.tool_call_chunks?.[0]?.id).toBeUndefined()

    const [initial, ...rest] = messages
    if (!initial) throw new Error('WebLLM returned no message chunks.')
    const result = rest.reduce((combined, message) => combined.concat(message), initial)
    expect(result.tool_calls).toMatchObject([
      { id: 'call-1', name: 'lookup', args: { code: 'A7' } },
    ])
    expect(result.response_metadata.finish_reason).toBe('tool_calls')
  })

  it('streams Hermes tool arguments before WebLLM provides a call ID', async () => {
    const yielded = pause()
    const resume = pause()
    async function* source() {
      yielded.resume()
      yield chunk({
        tool_calls: [
          { index: 0, type: 'function', function: { name: 'lookup', arguments: '{"code":' } },
        ],
      })
      await resume.wait
      yield chunk({
        tool_calls: [{ index: 0, type: 'function', function: { arguments: '"A7"}' } }],
      })
      yield chunk({}, 'tool_calls')
    }

    const stream = readChunks(nativeChunks(source(), true), 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC')
    const pending = stream.next()
    await yielded.wait
    const streamedBeforeResume = await Promise.race([
      pending.then(() => true),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 0)),
    ])
    resume.resume()
    const first = await pending
    const messages = first.done ? [] : [first.value]
    for await (const item of stream) messages.push(item)

    expect(streamedBeforeResume).toBe(true)
    const firstMessage = messages[0]?.message
    if (!AIMessageChunk.isInstance(firstMessage))
      throw new Error('WebLLM returned no AI message chunks.')
    expect(firstMessage.tool_call_chunks).toEqual([
      expect.objectContaining({ index: 0, name: 'lookup', args: '{"code":' }),
    ])
    const [initial, ...rest] = messages
    if (!initial || !AIMessageChunk.isInstance(initial.message))
      throw new Error('WebLLM returned no AI message chunks.')
    const result = rest.reduce((combined, item) => {
      if (!AIMessageChunk.isInstance(item.message)) throw new Error('Expected an AI message.')
      return combined.concat(item.message)
    }, initial.message)
    expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expect(result.tool_calls?.[0]?.id).toEqual(expect.any(String))
  })

  it('does not complete an id-less native call truncated at the token limit', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [
            { index: 0, type: 'function', function: { name: 'lookup', arguments: '{"code":' } },
          ],
        },
        'length',
      ),
    )
    const messages: AIMessageChunk[] = []
    for await (const message of await model.bindTools([lookup]).stream('Look up A7.'))
      messages.push(message)

    const [initial, ...rest] = messages
    if (!initial) throw new Error('WebLLM returned no message chunks.')
    const result = rest.reduce((combined, message) => combined.concat(message), initial)

    expect(result.tool_call_chunks).toMatchObject([{ index: 0, name: 'lookup', args: '{"code":' }])
    expect(result.tool_call_chunks?.[0]?.id).toBeUndefined()
    expect(result.tool_calls).toEqual([])
    expect(result.response_metadata.finish_reason).toBe('length')
  })

  it('keeps out-of-order Hermes calls ordered when WebLLM omits IDs', async () => {
    const { model } = setup(
      chunk({
        tool_calls: [
          { index: 1, type: 'function', function: { name: 'lookup', arguments: '{"code":' } },
        ],
      }),
      chunk({
        tool_calls: [
          {
            index: 0,
            type: 'function',
            function: { name: 'lookup', arguments: '{"code":"A7"}' },
          },
          { index: 1, type: 'function', function: { arguments: '"A8"}' } },
        ],
      }),
      chunk({}, 'tool_calls'),
    )
    const messages: AIMessageChunk[] = []
    for await (const message of await model.bindTools([lookup]).stream('Look up A7 and A8.'))
      messages.push(message)

    const [initial, ...rest] = messages
    if (!initial) throw new Error('WebLLM returned no message chunks.')
    const result = rest.reduce((combined, message) => combined.concat(message), initial)

    expect(result.tool_calls).toMatchObject([
      { name: 'lookup', args: { code: 'A7' } },
      { name: 'lookup', args: { code: 'A8' } },
    ])
    expect(new Set(result.tool_calls?.map((call) => call.id)).size).toBe(2)
  })

  it('streams native tool-call events before generation finishes', async () => {
    const { model, create } = setup()
    const yielded = pause()
    const resume = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yielded.resume()
        yield chunk({
          tool_calls: [
            { index: 0, id: 'call-1', function: { name: 'look', arguments: '{"code":' } },
          ],
        })
        await resume.wait
        yield chunk({
          tool_calls: [{ index: 0, function: { name: 'up', arguments: '"A7"}' } }],
        })
        yield chunk({}, 'tool_calls')
      })(),
    )

    const stream = model.streamEvents('Look up A7.', { tools: [lookup] })
    const events: ChatModelStreamEvent[] = []
    const toolCalls = stream.toolCalls
    const consuming = (async () => {
      for await (const event of stream) events.push(event)
    })()
    try {
      await yielded.wait
      await vi.waitFor(
        () =>
          expect(
            events.some(
              (event) =>
                event.event === 'content-block-start' && event.content.type === 'tool_call_chunk',
            ),
          ).toBe(true),
        { timeout: 5000, interval: 10 },
      )
      const prematureCall = await Promise.race([
        toolCalls.then(() => ({ ready: true as const })),
        new Promise<{ ready: false }>((resolve) => setTimeout(() => resolve({ ready: false }), 0)),
      ])
      expect(prematureCall).toEqual({ ready: false })
    } finally {
      resume.resume()
    }
    const [, calls] = await Promise.all([consuming, toolCalls])

    expect(events).toContainEqual(
      expect.objectContaining({
        event: 'content-block-delta',
        delta: { type: 'block-delta', fields: { type: 'tool_call_chunk', args: '{"code":' } },
      }),
    )
    expect(events).toContainEqual(
      expect.objectContaining({
        event: 'content-block-delta',
        delta: {
          type: 'block-delta',
          fields: { type: 'tool_call_chunk', name: 'lookup', args: '{"code":"A7"}' },
        },
      }),
    )
    expect(events).toContainEqual(
      expect.objectContaining({
        event: 'content-block-finish',
        content: { type: 'tool_call', id: 'call-1', name: 'lookup', args: { code: 'A7' } },
      }),
    )
    expect(calls).toMatchObject([{ id: 'call-1', name: 'lookup', args: { code: 'A7' } }])
  })

  it('waits for the stop reason before publishing a complete-looking tool call', async () => {
    const { model, create } = setup()
    const yielded = pause()
    const resume = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yielded.resume()
        yield chunk({
          tool_calls: [
            { index: 0, id: 'call-1', function: { name: 'lookup', arguments: '{"code":"A7"}' } },
          ],
        })
        await resume.wait
        yield chunk({}, 'tool_calls')
      })(),
    )

    const stream = model.streamEvents('Look up A7.', { tools: [lookup] })
    const toolCalls = stream.toolCalls
    let started = false
    const consuming = (async () => {
      for await (const event of stream) {
        if (
          event.event === 'content-block-delta' &&
          event.delta.type === 'block-delta' &&
          'args' in event.delta.fields
        )
          started = true
      }
    })()

    try {
      await yielded.wait
      await vi.waitFor(() => expect(started).toBe(true), { timeout: 5000, interval: 10 })
      const premature = await Promise.race([
        toolCalls.then(() => true),
        new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 0)),
      ])
      expect(premature).toBe(false)
    } finally {
      resume.resume()
    }
    const [, calls] = await Promise.all([consuming, toolCalls])

    expect(calls).toMatchObject([{ id: 'call-1', name: 'lookup', args: { code: 'A7' } }])
  })

  it('rejects a changed tool call ID without finishing the partial call', async () => {
    const { model, create } = setup()
    const yielded = pause()
    const resume = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yielded.resume()
        yield chunk({
          tool_calls: [
            { index: 0, id: 'call-1', function: { name: 'lookup', arguments: '{"code":"A7"}' } },
          ],
        })
        await resume.wait
        yield chunk(
          { tool_calls: [{ index: 0, id: 'call-2', function: { arguments: '' } }] },
          'tool_calls',
        )
      })(),
    )

    const stream = model.streamEvents('Look up A7.', { tools: [lookup] })
    const events: ChatModelStreamEvent[] = []
    const toolCalls = stream.toolCalls
    const started = pause()
    const consuming = (async () => {
      for await (const event of stream) {
        events.push(event)
        if (
          event.event === 'content-block-delta' &&
          event.delta.type === 'block-delta' &&
          'args' in event.delta.fields
        )
          started.resume()
      }
    })()

    await yielded.wait
    await started.wait
    resume.resume()

    await expect(Promise.all([consuming, toolCalls])).rejects.toThrow('changed a tool call ID')
    expect(
      events.some(
        (event) => event.event === 'content-block-finish' && event.content.type === 'tool_call',
      ),
    ).toBe(false)
  })

  it('rejects native tool calls with duplicate IDs', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [
            { index: 0, id: 'same-id', function: { name: 'lookup', arguments: '{}' } },
            { index: 1, id: 'same-id', function: { name: 'lookup', arguments: '{}' } },
          ],
        },
        'tool_calls',
      ),
    )

    await expect(model.bindTools([lookup]).invoke('Look up both codes.')).rejects.toThrow(
      'duplicate tool call IDs',
    )
  })

  it('emits native serialized tool JSON only as a tool call', async () => {
    const raw = '[{"name":"lookup","arguments":{"code":"A7"}}]'
    const { model } = setup(
      ...raw.split('').map((content) => chunk({ content })),
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: '{"code":"A7"}' } }] },
        'tool_calls',
      ),
    )

    const result = await model.bindTools([lookup]).invoke('Look up A7.')

    expect(result.text).toBe('')
    expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
  })

  it('preserves extra native JSON fields beside the parsed tool call', async () => {
    const raw = '[{"name":"lookup","arguments":{"code":"A7"},"note":"Keep this note."}]'
    const { model } = setup(
      ...raw.split('').map((content) => chunk({ content })),
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: '{"code":"A7"}' } }] },
        'tool_calls',
      ),
    )

    const result = await model.bindTools([lookup]).invoke('Look up A7.')

    expect(result.text).toBe(raw)
    expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
  })

  it.each(['1e999', '-1e999'])(
    'rejects native arguments changed to null by overflow: %s',
    async (number) => {
      const raw = `[{"name":"lookup","arguments":{"code":"A7","score":${number}}}]`
      const args = JSON.stringify(JSON.parse(raw)[0].arguments)
      const { model } = setup(
        chunk({ content: raw }),
        chunk(
          { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: args } }] },
          'tool_calls',
        ),
      )

      await expect(model.bindTools([lookup]).invoke('Look up A7.')).rejects.toThrow(
        'finite numbers',
      )
    },
  )

  it('preserves incomplete native tool JSON at the token limit', async () => {
    const raw = '[{"name":"lookup","arguments":{'
    const { model } = setup(
      chunk({ content: raw.slice(0, 12) }),
      chunk({ content: raw.slice(12) }, 'length'),
    )
    const result = await model.bindTools([lookup]).invoke('Look up A7.')
    expect(result.text).toBe(raw)
    expect(result.tool_calls).toEqual([])
  })

  it('preserves native tool text before reporting a cancelled generation', async () => {
    const raw = '[{"name":"lookup","arguments":{'
    const { model } = setup(chunk({ content: raw }, 'abort'))
    const output: string[] = []
    await expect(
      (async () => {
        for await (const item of await model.bindTools([lookup]).stream('Look up A7.'))
          output.push(item.text)
      })(),
    ).rejects.toThrow('cancelled')
    expect(output.join('')).toBe(raw)
  })

  it.each(['["A7"]', '[1e999]'])(
    'preserves unrelated JSON beside a native call: %s',
    async (text) => {
      const { model } = setup(
        chunk({ content: text }),
        chunk(
          { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: '{"code":"A7"}' } }] },
          'tool_calls',
        ),
      )
      const result = await model.bindTools([lookup]).invoke('Look up A7.')
      expect(result.text).toBe(text)
      expect(result.tool_calls).toHaveLength(1)
    },
  )
})
