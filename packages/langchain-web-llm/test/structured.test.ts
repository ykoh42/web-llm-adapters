import { AIMessage } from '@langchain/core/messages'
import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import { chunk, setup } from '../fixtures/engine.js'
import { ChatWebLLM } from '../src/index.js'

function response(args: string) {
  return setup(
    chunk(
      {
        tool_calls: [
          {
            index: 0,
            id: 'call',
            type: 'function',
            function: { name: 'extract', arguments: args },
          },
        ],
      },
      'tool_calls',
    ),
  )
}
describe('structured output', () => {
  it('uses LangChain schema validation and reports parser failures', async () => {
    const { model } = response('{"answer":"wrong type"}')
    await expect(
      model.withStructuredOutput(z.object({ answer: z.number() })).invoke('Hi'),
    ).rejects.toThrow()
    const raw = await model
      .withStructuredOutput(z.object({ answer: z.number() }), { includeRaw: true })
      .invoke('Hi')
    expect(AIMessage.isInstance(raw.raw) && raw.raw.tool_calls?.[0]?.args).toEqual({
      answer: 'wrong type',
    })
    expect(raw.parsed).toBe(null)
  })
  it('reports the structured-output schema to LangChain tracing', async () => {
    const { model } = response('{"answer":42}')
    let options: unknown
    const schema = z.object({ answer: z.number() })
    const result = await model.withStructuredOutput(schema).invoke('Hi', {
      callbacks: [
        {
          handleChatModelStart(_model, _messages, _id, _parent, extra) {
            options = extra?.options
          },
        },
      ],
    })
    expect(result).toEqual({ answer: 42 })
    expect(options).toMatchObject({
      ls_structured_output_format: {
        kwargs: { method: 'functionCalling' },
        schema: { type: 'object', properties: { answer: { type: 'number' } } },
      },
    })
  })
  it.each(['jsonMode', 'jsonSchema'])(
    'uses native %s without synthetic tool calls',
    async (method) => {
      const { model, create } = setup(chunk({ content: '{"answer":42}' }, 'stop'))
      expect(
        await model.withStructuredOutput(z.object({ answer: z.number() }), { method }).invoke('Hi'),
      ).toEqual({ answer: 42 })
      const request = create.mock.calls[0]?.[0]
      expect(request?.response_format?.type).toBe('json_object')
      expect(request).not.toHaveProperty('tools')
      if (method === 'jsonSchema') expect(request?.response_format?.schema).toContain('answer')
    },
  )
  it('rejects strict mode and invalid methods before starting the engine', () => {
    const { model, create } = setup()
    expect(() =>
      model.withStructuredOutput(z.object({ answer: z.number() }), { strict: true }),
    ).toThrow('strict')
    expect(() => model.withStructuredOutput({ type: 'object' }, { method: 'invalid' })).toThrow(
      'functionCalling',
    )
    expect(create).not.toHaveBeenCalled()
  })
  it('rejects constrained Gemma output with thinking enabled', async () => {
    const { engine } = setup()
    const model = new ChatWebLLM({
      engine,
      modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
      request: { extra_body: { enable_thinking: true } },
    })
    await expect(
      model
        .withStructuredOutput(z.object({ answer: z.number() }), { method: 'jsonSchema' })
        .invoke('Hi'),
    ).rejects.toThrow('does not support')
  })
})
