import { describe, expect, it } from 'vitest'
import type { ResponseFormat } from '@mlc-ai/web-llm'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { clampMaxTokensToContext } from '@earendil-works/pi-ai/api/simple-options'
import { setup } from './helpers.js'

const context = { messages: [{ role: 'user' as const, content: 'Hello', timestamp: 0 }] }

describe('Pi sampling options', () => {
  it.each([
    ['deferred request', { deferred: true }, 'WebLLM does not support deferred requests.'],
    [
      'deferred window',
      { deferred: { window: '1h' } },
      'WebLLM does not support deferred requests.',
    ],
    [
      'thinking budget',
      { thinkingBudgets: { high: 100 } },
      'WebLLM does not support thinking token budgets.',
    ],
  ] as const)(
    'rejects unsupported %s before opening WebLLM',
    async (_name, options, errorMessage) => {
      const { models, model, create } = setup()
      const results = await Promise.all([
        models.complete(model, context, options),
        models.completeSimple(model, context, options),
      ])

      for (const result of results)
        expect(result).toMatchObject({ stopReason: 'error', errorMessage })
      expect(create).not.toHaveBeenCalled()
    },
  )

  it('accepts empty thinking budgets', async () => {
    const { models, model, create } = setup()
    const result = await models.complete(model, context, { thinkingBudgets: {} })

    expect(result.stopReason).toBe('stop')
    expect(create.mock.calls[0]?.[0].extra_body).toBeUndefined()
  })

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

  it('uses the same supported default level for sampling and thinking', async () => {
    const { models, model, create } = setup(undefined, { reasoning: true })
    model.thinkingLevelMap = { off: null }
    model.samplingParamsByThinkingLevel = {
      off: { temperature: 0.1 },
      minimal: { temperature: 0.8 },
    }
    const before = structuredClone(model)
    await models.completeSimple(model, context)

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: 0.8,
      extra_body: { enable_thinking: true },
    })
    expect(model).toEqual(before)
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

describe('Pi output token limits', () => {
  it('clamps only simple calls to the remaining context', async () => {
    const { models, model, create } = setup()
    const longContext = normalizeContext({
      messages: [{ role: 'user' as const, content: 'x'.repeat(8_000), timestamp: 0 }],
    })
    const expected = clampMaxTokensToContext(model, longContext, model.maxTokens)

    await models.complete(model, longContext)
    expect(create.mock.calls[0]?.[0].max_tokens).toBe(model.maxTokens)

    await models.completeSimple(model, longContext)
    expect(create.mock.calls[1]?.[0].max_tokens).toBe(expected)

    await models.completeSimple(model, longContext, { samplingParams: { max_tokens: 73 } })
    expect(create.mock.calls[2]?.[0].max_tokens).toBe(73)
  })

  it('preserves a native null cap on long simple calls', async () => {
    const { models, model, create } = setup(undefined, { request: { max_tokens: null } })
    const longContext = normalizeContext({
      messages: [{ role: 'user' as const, content: 'x'.repeat(8_000), timestamp: 0 }],
    })
    await models.completeSimple(model, longContext)

    expect(create.mock.calls[0]?.[0].max_tokens).toBeNull()
  })

  it.each([false, true])('preserves a native null default (simple=%s)', async (simple) => {
    const { models, model, create } = setup(undefined, {
      maxTokens: 1,
      request: { max_tokens: null },
    })
    const result = simple
      ? await models.completeSimple(model, context)
      : await models.complete(model, context)

    expect(result.stopReason).toBe('stop')
    expect(create.mock.calls[0]?.[0].max_tokens).toBeNull()
  })

  it.each(['request', 'samplingParams', 'model', 'level'] as const)(
    'preserves native null from %s with the usual option precedence',
    async (source) => {
      const { models, model, create } = setup(undefined, { request: { max_tokens: 64 } })
      if (source === 'model') model.samplingParams = { max_tokens: null }
      if (source === 'level') model.samplingParamsByThinkingLevel = { off: { max_tokens: null } }
      const result = await models.complete(model, context, {
        ...(source !== 'request' && { maxTokens: 32 }),
        ...(source === 'request' && { request: { max_tokens: null } }),
        ...(source === 'samplingParams' && { samplingParams: { max_tokens: null } }),
      })

      expect(result.stopReason).toBe('stop')
      expect(create.mock.calls[0]?.[0].max_tokens).toBeNull()
    },
  )

  it.each(['defaults', 'request'])(
    'lets named maxTokens override a native null from %s',
    async (source) => {
      const { models, model, create } = setup(undefined, {
        request: { max_tokens: source === 'defaults' ? null : 64 },
      })
      await models.complete(model, context, {
        maxTokens: 32,
        ...(source === 'request' && { request: { max_tokens: null } }),
      })

      expect(create.mock.calls[0]?.[0].max_tokens).toBe(32)
    },
  )

  it('inherits the native default when a JavaScript call contains undefined', async () => {
    const { models, model, create } = setup(undefined, { request: { max_tokens: 64 } })
    const request = { max_tokens: 32 }
    Object.assign(request, { max_tokens: undefined })
    await models.complete(model, context, { request })

    expect(create.mock.calls[0]?.[0].max_tokens).toBe(64)
  })

  for (const simple of [false, true]) {
    it.each(['samplingParams', 'model', 'level'] as const)(
      `preserves an unset native token limit from %s (simple=${simple})`,
      async (source) => {
        const { models, model, create } = setup(undefined, { request: { max_tokens: 64 } })
        if (source === 'model') model.samplingParams = { max_tokens: undefined }
        if (source === 'level')
          model.samplingParamsByThinkingLevel = { off: { max_tokens: undefined } }
        const options = {
          maxTokens: 32,
          ...(source === 'samplingParams' && { samplingParams: { max_tokens: undefined } }),
        }
        const result = simple
          ? await models.completeSimple(model, context, options)
          : await models.complete(model, context, options)

        expect(result.stopReason).toBe('stop')
        expect(result.content).toEqual([{ type: 'text', text: 'Hello' }])
        expect(create).toHaveBeenCalledExactlyOnceWith(
          expect.objectContaining({ max_tokens: undefined }),
        )
      },
    )
  }

  it.each([0, -1, NaN, Infinity, 1.5])('still rejects an invalid cap %s', async (max_tokens) => {
    const { models, model, create } = setup()
    const result = await models.complete(model, context, { request: { max_tokens } })

    expect(result.stopReason).toBe('error')
    expect(result.errorMessage).toContain('positive integer')
    expect(create).not.toHaveBeenCalled()
  })
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

describe.each([false, true])('native thinking sampling (simple=%s)', (simple) => {
  for (const source of ['model', 'level', 'call'] as const) {
    it.each([false, true, null, undefined])(
      `preserves enable_thinking=%s from ${source} sampling`,
      async (enable_thinking) => {
        const { models, model, create } = setup(undefined, { reasoning: true })
        const reasoning = enable_thinking === true ? undefined : ('high' as const)
        const extra_body = { enable_thinking, enable_latency_breakdown: true }
        if (source === 'model') model.samplingParams = { extra_body }
        if (source === 'level')
          model.samplingParamsByThinkingLevel = { [reasoning ?? 'off']: { extra_body } }
        const options = {
          ...(reasoning !== undefined && { reasoning }),
          ...(source === 'call' && { samplingParams: { extra_body } }),
        } as const
        const before = structuredClone({ model, options })
        const result = simple
          ? await models.completeSimple(model, context, options)
          : await models.complete(model, context, options)

        expect(result.stopReason).toBe('stop')
        expect(create.mock.calls[0]?.[0].extra_body).toEqual(extra_body)
        expect({ model, options }).toEqual(before)
      },
    )
  }

  it('forwards an explicit sampling body without adding thinking fields', async () => {
    const { models, model, create } = setup(undefined, { reasoning: true })
    const options = {
      reasoning: 'high' as const,
      samplingParams: { extra_body: { enable_latency_breakdown: true } },
    }
    if (simple) await models.completeSimple(model, context, options)
    else await models.complete(model, context, options)
    expect(create.mock.calls[0]?.[0].extra_body).toEqual({ enable_latency_breakdown: true })
  })

  it('keeps named reasoning above native defaults when sampling does not replace the body', async () => {
    const { models, model, create } = setup(undefined, {
      reasoning: true,
      request: { extra_body: { enable_thinking: false, enable_latency_breakdown: true } },
    })
    const options = { reasoning: 'high' as const, samplingParams: { temperature: 0.7 } }
    if (simple) await models.completeSimple(model, context, options)
    else await models.complete(model, context, options)
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: 0.7,
      extra_body: { enable_thinking: true, enable_latency_breakdown: true },
    })
  })
})
