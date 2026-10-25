import { describe, expect, it, vi } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { chunk, setup } from './helpers.js'

const context = { messages: [{ role: 'user' as const, content: 'Hello', timestamp: 0 }] }
const json = { type: 'json_object' as const, schema: '{}' }
const text = { type: 'text' as const }

describe('Pi payload format changes', () => {
  describe.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s thinking stops',
    (modelId) => {
      it.each([
        { initial: [], selected: ['DONE'], stopReason: 'stop' },
        { initial: ['DONE'], selected: [], stopReason: 'error' },
      ])('honors stop strings selected by onPayload: $initial to $selected', async (test) => {
        const { models, model, create } = setup([chunk({ content: '<think>Plan' }, 'stop')], {
          modelId,
          request: { stop: test.initial, extra_body: { enable_thinking: true } },
        })
        const response = await models.complete(model, context, {
          onPayload: (payload) => {
            if (!payload || typeof payload !== 'object')
              throw new Error('Expected a native request.')
            return { ...payload, stop: test.selected }
          },
        })

        expect(create.mock.calls[0]?.[0].stop).toEqual(test.selected)
        expect(response).toMatchObject({
          stopReason: test.stopReason,
          content: [{ type: 'thinking', thinking: 'Plan' }],
        })
        if (test.stopReason === 'error')
          expect(response.errorMessage).toBe('Qwen ended inside a thinking block.')
      })
    },
  )

  describe.each(['Qwen3-8B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])('%s framing', (modelId) => {
    it.each([
      {
        initial: text,
        selected: json,
        output: '{"text":"<think>literal</think>"}',
        expected: [{ type: 'text', text: '{"text":"<think>literal</think>"}' }],
      },
      {
        initial: json,
        selected: text,
        output: '<think>Plan</think>Answer',
        expected: [
          { type: 'thinking', thinking: 'Plan' },
          { type: 'text', text: 'Answer' },
        ],
      },
    ])('parses the format selected by onPayload: $initial.type to $selected.type', async (test) => {
      const { models, model, create } = setup([chunk({ content: test.output }, 'stop')], {
        modelId,
        request: { response_format: test.initial },
      })
      const response = await models.complete(model, context, {
        onPayload: (payload) => {
          if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
          return { ...payload, response_format: test.selected }
        },
      })

      expect(create.mock.calls[0]?.[0].response_format).toEqual(test.selected)
      expect(response).toMatchObject({ stopReason: 'stop', content: test.expected })
    })
  })

  it.each([
    { initial: true, selected: false, expected: '{"answer":42}' },
    { initial: false, selected: true, expected: '<think>\n\n</think>\n\n{"answer":42}' },
  ])('uses onPayload thinking $selected when separating a structured prefix', async (test) => {
    const output = '<think>\n\n</think>\n\n{"answer":42}'
    const { models, model, create } = setup([chunk({ content: output }, 'stop')], {
      modelId: 'Qwen3.5-4B-q4f16_1-MLC',
      request: { response_format: json, extra_body: { enable_thinking: test.initial } },
    })
    const response = await models.complete(model, context, {
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, extra_body: { enable_thinking: test.selected } }
      },
    })
    expect(create.mock.calls[0]?.[0].extra_body?.enable_thinking).toBe(test.selected)
    expect(response).toMatchObject({
      stopReason: 'stop',
      content: [{ type: 'text', text: test.expected }],
    })
  })

  it.each([
    { initial: [], selected: ['END'], stopReason: 'stop' },
    { initial: ['END'], selected: [], stopReason: 'error' },
  ])('uses Gemma stop strings selected by onPayload: $initial to $selected', async (test) => {
    const { models, model, engine } = setup(undefined, {
      modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
      request: { stop: test.initial },
    })
    const create = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield {
          id: 'raw',
          object: 'text_completion' as const,
          model: model.id,
          created: 0,
          choices: [{ index: 0, text: '<|channel>thought\nPlan', finish_reason: 'stop' as const }],
        }
      })(),
    )
    const response = await models.complete(model, context, {
      reasoning: 'low',
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, stop: test.selected }
      },
    })

    expect(create.mock.calls[0]?.[0].stop).toEqual(test.selected)
    expect(response).toMatchObject({
      stopReason: test.stopReason,
      content: [{ type: 'thinking', thinking: 'Plan' }],
    })
  })

  it.each([
    { ignore_eos: true },
    { response_format: { type: 'json_object', schema: '{}' } },
    { extra_body: { resumable: { enabled: true } } },
  ])('keeps Gemma protocol limits after onPayload: %j', async (override) => {
    const { models, model, engine } = setup(undefined, {
      modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    })
    const create = vi.spyOn(engine.completions, 'create')
    const response = await models.complete(model, context, {
      reasoning: 'low',
      onPayload: (payload) => {
        if (!payload || typeof payload !== 'object') throw new Error('Expected a native request.')
        return { ...payload, ...override }
      },
    })

    expect(response).toMatchObject({
      stopReason: 'error',
      errorMessage:
        'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
    })
    expect(create).not.toHaveBeenCalled()
  })

  it.each(['Qwen3-8B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s rejects a structured format added to a tool request by onPayload',
    async (modelId) => {
      const { models, model, create } = setup(undefined, { modelId })
      const response = await models.complete(
        model,
        {
          ...context,
          tools: [{ name: 'lookup', description: 'Look up a code.', parameters: Type.Object({}) }],
        },
        {
          onPayload: (payload) => {
            if (!payload || typeof payload !== 'object')
              throw new Error('Expected a native request.')
            return { ...payload, response_format: json }
          },
        },
      )

      expect(response).toMatchObject({
        stopReason: 'error',
        errorMessage: 'Qwen tool calls cannot use a structured response format.',
      })
      expect(create).not.toHaveBeenCalled()
    },
  )
})

for (const modelId of ['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  it.each([false, true])(
    `${modelId} follows thinking=%s selected by onPayload on a prefilled engine`,
    async (enabled) => {
      const source = enabled ? 'Plan.\n</think>42' : '<think>\n\n</think>42'
      const { models, model, create } = setup([chunk({ content: source }, 'stop')], {
        modelId,
        thinkingPrefilled: true,
        request: { extra_body: { enable_thinking: !enabled } },
      })
      const response = await models.complete(model, context, {
        onPayload: (request) => {
          if (!request || typeof request !== 'object') throw new Error('Expected a request.')
          return { ...request, extra_body: { enable_thinking: enabled } }
        },
      })
      expect(response.stopReason).toBe('stop')
      expect(
        response.content
          .filter((part) => part.type === 'thinking')
          .map((part) => part.thinking)
          .join(''),
      ).toBe(enabled ? 'Plan.\n' : '\n\n')
      expect(
        response.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join(''),
      ).toBe('42')
      expect(create.mock.calls[0]?.[0].extra_body?.enable_thinking).toBe(enabled)
    },
  )
}
