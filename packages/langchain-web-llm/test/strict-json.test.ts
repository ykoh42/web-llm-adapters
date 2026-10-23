import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

const schema = {
  type: 'object',
  properties: { answer: { type: 'integer', const: 42 } },
  required: ['answer'],
  additionalProperties: false,
}

describe.each(['local', 'browser'])('%s strict JSON', (transport) => {
  function model(modelId = 'Qwen3.5-4B-q4f16_1-MLC') {
    const native = setup(chunk({ content: '{"answer":42}' }, 'stop'))
    const bridge = createBrowserBridge(native.engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        const result = await run(structuredClone(arg))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    return {
      ...native,
      bridge,
      model:
        transport === 'browser'
          ? new BrowserChatWebLLM(page, { modelId })
          : new ChatWebLLM({ engine: native.engine, modelId }),
    }
  }

  for (const includeRaw of [false, true] as const) {
    it.each([
      { modelId: 'Qwen3.5-4B-q4f16_1-MLC', method: 'jsonSchema' },
      { modelId: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', method: undefined },
    ])(`uses the native schema for $modelId (includeRaw=${includeRaw})`, async (options) => {
      const state = model(options.modelId)
      try {
        for (const strict of [undefined, false, true]) {
          const config = {
            ...(options.method !== undefined && { method: options.method }),
            ...(strict !== undefined && { strict }),
          }
          const runnable = includeRaw
            ? state.model.withStructuredOutput<{ answer: number }>(schema, {
                ...config,
                includeRaw: true,
              })
            : state.model.withStructuredOutput<{ answer: number }>(schema, config)
          const result = await runnable.invoke('Return JSON with answer 42.')
          if ('parsed' in result) {
            expect(result.parsed).toEqual({ answer: 42 })
            expect(result.raw.text).toBe('{"answer":42}')
          } else expect(result).toEqual({ answer: 42 })
        }
        const requests = state.create.mock.calls.map(([request]) => request)
        expect(requests).toHaveLength(3)
        expect(requests[1]).toEqual(requests[0])
        expect(requests[2]).toEqual(requests[0])
        expect(requests[0]).toMatchObject({
          messages: [{ role: 'user', content: 'Return JSON with answer 42.' }],
          response_format: { type: 'json_object', schema: JSON.stringify(schema) },
        })
        expect(requests[0]).not.toHaveProperty('tools')
      } finally {
        await state.bridge.close()
      }
    })
  }

  it.each(['jsonMode', 'functionCalling'])(
    'rejects strict %s without native schema constraints',
    async (method) => {
      const state = model()
      try {
        expect(() => state.model.withStructuredOutput(schema, { method, strict: true })).toThrow(
          'strict',
        )
        expect(state.create).not.toHaveBeenCalled()
      } finally {
        await state.bridge.close()
      }
    },
  )

  it('keeps native schema errors without retrying or changing methods', async () => {
    const state = model()
    state.create.mockRejectedValue(new Error('Native schema compilation failed.'))
    try {
      await expect(
        state.model
          .withStructuredOutput(schema, { method: 'jsonSchema', strict: true })
          .invoke('Return JSON with answer 42.'),
      ).rejects.toThrow('Native schema compilation failed.')
      expect(state.create).toHaveBeenCalledOnce()
    } finally {
      await state.bridge.close()
    }
  })
})
