import { describe, expect, it } from 'vitest'
import type { ResponseFormat } from '@mlc-ai/web-llm'
import { setup } from './helpers.js'

const context = { messages: [{ role: 'user' as const, content: 'Hello', timestamp: 0 }] }

describe('Pi sampling options', () => {
  it.each([false, true])(
    'preserves native requests and output without thinking-level settings (simple=%s)',
    async (simple) => {
      const { models, model, create } = setup(undefined, {
        reasoning: true,
        request: {
          temperature: 0.1,
          max_tokens: 70,
          seed: 7,
          extra_body: { enable_thinking: true },
        },
      })
      model.samplingParams = { temperature: 0.5, top_p: 0.9 }
      const options = {
        temperature: 0.3,
        maxTokens: 50,
        samplingParams: { temperature: 0.7, max_tokens: 80 },
      }
      const result = simple
        ? await models.completeSimple(model, context, options)
        : await models.complete(model, context, options)

      expect(create.mock.calls[0]?.[0]).toEqual({
        model: model.id,
        messages: [{ role: 'user', content: 'Hello' }],
        n: 1,
        stream: true,
        stream_options: { include_usage: true },
        temperature: 0.7,
        max_tokens: 80,
        top_p: 0.9,
        seed: 7,
        extra_body: { enable_thinking: !simple },
      })
      expect(result).toMatchObject({
        stopReason: 'stop',
        content: [{ type: 'text', text: 'Hello' }],
      })
    },
  )

  it.each([
    { reasoning: undefined, temperature: 0.2 },
    { reasoning: 'high' as const, temperature: 0.9 },
  ])('applies $reasoning sampling without changing model defaults', async (options) => {
    const { models, model, create } = setup(undefined, { reasoning: true })
    model.samplingParams = { temperature: 0.7, top_p: 0.8 }
    model.samplingParamsByThinkingLevel = {
      off: { temperature: 0.2 },
      high: { temperature: 0.9 },
    }
    const before = structuredClone(model)
    await models.completeSimple(model, context, {
      ...(options.reasoning && { reasoning: options.reasoning }),
    })

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: options.temperature,
      top_p: 0.8,
      extra_body: { enable_thinking: options.reasoning !== undefined },
    })
    expect(model).toEqual(before)
  })

  it('selects sampling for the supported thinking level after clamping', async () => {
    const { models, model, create } = setup(undefined, { reasoning: true })
    model.thinkingLevelMap = { high: null }
    model.samplingParamsByThinkingLevel = {
      high: { temperature: 0.9 },
      medium: { temperature: 0.6 },
    }
    await models.completeSimple(model, context, { reasoning: 'high' })

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: 0.6,
      extra_body: { enable_thinking: true },
    })
  })

  it('disables thinking when an explicit level is clamped to off', async () => {
    const { models, model, create } = setup(undefined, { reasoning: true })
    model.thinkingLevelMap = {
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    }
    model.samplingParamsByThinkingLevel = {
      off: { temperature: 0.2 },
      high: { temperature: 0.9 },
    }
    await models.completeSimple(model, context, { reasoning: 'high' })

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: 0.2,
      extra_body: { enable_thinking: false },
    })
  })

  it('still rejects an explicit thinking request on a model without reasoning', async () => {
    const { models, model, create } = setup(undefined, { reasoning: false })
    const result = await models.completeSimple(model, context, { reasoning: 'high' })

    expect(result).toMatchObject({
      stopReason: 'error',
      errorMessage: 'This WebLLM model does not support reasoning.',
    })
    expect(create).not.toHaveBeenCalled()
  })

  it('lets request sampling override thinking-level and named options', async () => {
    const { models, model, create } = setup(undefined, { reasoning: true })
    model.samplingParams = { temperature: 0.7, top_p: 0.8, presence_penalty: 0.5 }
    model.samplingParamsByThinkingLevel = {
      high: { temperature: 0.9, top_p: 0.95 },
    }
    await models.completeSimple(model, context, {
      reasoning: 'high',
      temperature: 0.1,
      samplingParams: { temperature: 0.4, top_p: 0.6 },
    })

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: 0.4,
      top_p: 0.6,
      presence_penalty: 0.5,
    })
  })

  it.each(['model', 'messages', 'prompt', 'tools', 'tool_choice', 'n', 'stream', 'stream_options'])(
    'rejects thinking-level overrides of %s before generation',
    async (key) => {
      const { models, model, create } = setup(undefined, { reasoning: true })
      model.samplingParamsByThinkingLevel = { high: { [key]: 'override' } }
      const result = await models.completeSimple(model, context, { reasoning: 'high' })

      expect(result).toMatchObject({
        stopReason: 'error',
        errorMessage: `Use the provider API instead of overriding ${key}.`,
      })
      expect(create).not.toHaveBeenCalled()
    },
  )
})

describe('Pi response formats', () => {
  it.each(['defaults', 'request', 'samplingParams'])(
    'supplies the JSON schema required by WebLLM for bare JSON mode from %s',
    async (source) => {
      const format = Object.freeze({ type: 'json_object' as const })
      const { models, model, create } = setup(undefined, {
        ...(source === 'defaults' && { request: { response_format: format } }),
      })
      await models.complete(model, context, {
        ...(source === 'request' && { request: { response_format: format } }),
        ...(source === 'samplingParams' && { samplingParams: { response_format: format } }),
      })

      expect(create.mock.calls[0]?.[0].response_format).toEqual({
        type: 'json_object',
        schema: '{}',
      })
      expect(format).toEqual({ type: 'json_object' })
    },
  )

  it('normalizes the response format selected after all sampling overrides', async () => {
    const original = { type: 'json_object' as const, schema: '{ "type": "object" }' }
    const selected = Object.freeze({ type: 'json_object' as const })
    const { models, model, create } = setup(undefined, { request: { response_format: original } })
    model.samplingParamsByThinkingLevel = { off: { response_format: selected } }
    await models.complete(model, context, {
      request: { response_format: { type: 'text' } },
    })

    expect(create.mock.calls[0]?.[0].response_format).toEqual({
      type: 'json_object',
      schema: '{}',
    })
    expect(original.schema).toBe('{ "type": "object" }')
    expect(selected).toEqual({ type: 'json_object' })
  })

  it.each([
    { type: 'json_object', schema: '{ "type": "object", "properties": {} }' },
    { type: 'grammar', grammar: 'root ::= "ok"' },
    {
      type: 'structural_tag',
      structural_tag: '{"type":"structural_tag","structures":[],"triggers":[]}',
    },
    { type: 'text' },
  ] satisfies ResponseFormat[])('preserves explicit $type format bytes', async (format) => {
    const before = structuredClone(format)
    Object.freeze(format)
    const { models, model, create } = setup()
    await models.complete(model, context, { request: { response_format: format } })

    expect(create.mock.calls[0]?.[0].response_format).toEqual(before)
    expect(format).toEqual(before)
  })
})
