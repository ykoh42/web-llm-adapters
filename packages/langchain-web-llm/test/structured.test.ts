import { AIMessage } from '@langchain/core/messages'
import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import { createAgent, toolStrategy } from 'langchain'
import { tool } from '@langchain/core/tools'
import { chunk, chunks, setup } from '../fixtures/engine.js'
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
  it('passes an agent response schema to WebLLM native JSON generation', async () => {
    const { engine, create } = setup(chunk({ content: '{"answer":42}' }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId: 'Qwen3.5-4B-q4f16_1-MLC' })
    const agent = createAgent({ model, responseFormat: z.object({ answer: z.number() }) })

    const result = await agent.invoke({ messages: [{ role: 'user', content: 'Six times seven?' }] })

    expect(result.structuredResponse).toEqual({ answer: 42 })
    const format = create.mock.calls[0]?.[0].response_format
    expect(format?.type).toBe('json_object')
    expect(JSON.parse(format?.schema ?? '')).toMatchObject({
      type: 'object',
      properties: { answer: { type: 'number' } },
      required: ['answer'],
    })
  })

  it('rejects Qwen agent native JSON combined with tools before inference', async () => {
    const { engine, create } = setup()
    const model = new ChatWebLLM({ engine, modelId: 'Qwen3.5-4B-q4f16_1-MLC' })
    const lookup = tool(() => 42, {
      name: 'lookup',
      description: 'Look up an answer.',
      schema: z.object({}),
    })
    const agent = createAgent({
      model,
      tools: [lookup],
      responseFormat: z.object({ answer: z.number() }),
    })
    await expect(
      agent.invoke({ messages: [{ role: 'user', content: 'Use lookup.' }] }),
    ).rejects.toThrow('cannot use a structured response format')
    expect(create).not.toHaveBeenCalled()
  })

  it('uses LangChain tool strategy through a real tool and a structured response', async () => {
    const { engine, create } = setup()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Use lookup.</think><tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
    )
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Format the answer.</think><tool_call><function=answer><parameter=answer>42</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
    )
    const model = new ChatWebLLM({
      engine,
      modelId: 'Qwen3.5-4B-q4f16_1-MLC',
      request: { extra_body: { enable_thinking: true } },
    })
    const inputs: unknown[] = []
    const lookup = tool(
      (input) => {
        inputs.push(input)
        return 42
      },
      { name: 'lookup', description: 'Look up an answer.', schema: z.object({ code: z.string() }) },
    )
    const agent = createAgent({
      model,
      tools: [lookup],
      responseFormat: toolStrategy({
        title: 'answer',
        type: 'object',
        properties: { answer: { type: 'number' } },
        required: ['answer'],
      }),
    })
    const result = await agent.invoke({
      messages: [{ role: 'user', content: 'Use lookup with code A7 and return its answer.' }],
    })
    expect(inputs).toEqual([{ code: 'A7' }])
    expect(result.structuredResponse).toEqual({ answer: 42 })
    expect(create).toHaveBeenCalledTimes(2)
    for (const [request] of create.mock.calls) {
      expect(request.response_format).toBeUndefined()
      expect(request.extra_body?.enable_thinking).toBe(true)
    }
  })

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
  it.each(['jsonMode', 'jsonSchema'])(
    'keeps the %s response format when call options change sampling',
    async (method) => {
      const { engine, create } = setup(chunk({ content: '{"answer":42}' }, 'stop'))
      const model = new ChatWebLLM({
        engine,
        modelId: 'test-model',
        maxTokens: 128,
        request: { seed: 7, response_format: { type: 'text' } },
      })

      const result = await model
        .withStructuredOutput(z.object({ answer: z.number() }), { method })
        .invoke('Six times seven?', { request: { temperature: 0 } })

      expect(result).toEqual({ answer: 42 })
      const request = create.mock.calls[0]?.[0]
      expect(request).toMatchObject({
        max_tokens: 128,
        seed: 7,
        temperature: 0,
        response_format: { type: 'json_object' },
      })
      if (method === 'jsonSchema') expect(request?.response_format?.schema).toContain('answer')
    },
  )

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    'rejects invalid numeric schema values without changing them to null: %s',
    (value) => {
      const { model, create } = setup()
      const schema = { type: 'object', properties: { value: { type: 'number', const: value } } }

      expect(() => model.withStructuredOutput(schema, { method: 'jsonSchema' })).toThrow(
        'finite numbers',
      )
      expect(create).not.toHaveBeenCalled()
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
