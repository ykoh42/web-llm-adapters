import { describe, expect, it, vi } from 'vitest'
import { ContextWindowOverflowError, ModelError } from '@strands-agents/sdk'
import { ModelError as ProtocolError } from '../../../../shared/web-llm/types.js'
import { modelError } from '../../src/engine/error.js'
import { chunk, collect, setup } from '../../fixtures/engine.js'

it('preserves a wrapped worker overflow as the original cause', async () => {
  const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
  const error = new Error('ContextWindowSizeExceededError: input is too long.')
  create.mockRejectedValueOnce(error)
  const failed = collect(model.stream([]))
  await expect(failed).rejects.toBeInstanceOf(ContextWindowOverflowError)
  await expect(failed).rejects.toHaveProperty('cause', error)
  await expect(collect(model.stream([]))).resolves.toContainEqual(
    expect.objectContaining({ type: 'modelMessageStopEvent', stopReason: 'endTurn' }),
  )
})

it('keeps an existing Strands overflow error unchanged', () => {
  const error = new ContextWindowOverflowError('ContextWindowSizeExceededError: input is too long.')
  expect(modelError(error)).toBe(error)
})

it('preserves an error proxy when its prototype cannot be inspected', async () => {
  const { model, create } = setup()
  const error = new Proxy(new Error('Native failed'), {
    getPrototypeOf() {
      throw new Error('Cannot inspect provider prototype')
    },
  })
  create.mockRejectedValueOnce(error)

  const failure = await collect(model.stream([])).then(
    () => undefined,
    (error: unknown) => error,
  )
  expect(failure).toBe(error)
})

it('recognizes a proxied native context overflow', async () => {
  const { model, create } = setup()
  const native = Object.assign(new Error('ContextWindowSizeExceededError: input is too long.'), {
    name: 'ContextWindowSizeExceededError',
  })
  const error = new Proxy(native, {
    getPrototypeOf() {
      throw new Error('Cannot inspect provider prototype')
    },
  })
  create.mockRejectedValueOnce(error)

  const failed = collect(model.stream([]))
  await expect(failed).rejects.toBeInstanceOf(ContextWindowOverflowError)
  await expect(failed).rejects.toHaveProperty(
    'message',
    'ContextWindowSizeExceededError: input is too long.',
  )
  const failure = await failed.catch((error: unknown) => error)
  expect((failure as Error).cause).toBe(error)
})

describe.each([
  'test-model',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
])('%s error boundary', (modelId) => {
  it.each([
    undefined,
    new Error('Native failed'),
    new ModelError('Provider failed', { cause: new Error('Original cause') }),
    new ContextWindowOverflowError('Context full'),
  ])('preserves the original rejection: %s', async (error) => {
    const { model, engine, create } = setup()
    model.updateConfig({ modelId, request: { extra_body: { enable_thinking: true } } })
    const native = modelId.startsWith('gemma-4-') ? vi.spyOn(engine.completions, 'create') : create
    native.mockRejectedValueOnce(error)

    await expect(collect(model.stream([]))).rejects.toBe(error)
  })

  it('maps a protocol error once and keeps its original cause', async () => {
    const { model, engine, create } = setup()
    model.updateConfig({ modelId, request: { extra_body: { enable_thinking: true } } })
    const native = modelId.startsWith('gemma-4-') ? vi.spyOn(engine.completions, 'create') : create
    const error = new ProtocolError('Invalid protocol')
    native.mockRejectedValueOnce(error)

    const failed = collect(model.stream([]))
    await expect(failed).rejects.toBeInstanceOf(ModelError)
    await expect(failed).rejects.toHaveProperty('message', error.message)
    await expect(failed).rejects.toHaveProperty('cause', error)
  })
})
