import { describe, expect, it } from 'vitest'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { ChatWebLLM } from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

const lookup = {
  type: 'function',
  function: {
    name: 'lookup',
    description: 'Look up a code.',
    parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] },
  },
}

describe('native WebLLM messages', () => {
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
