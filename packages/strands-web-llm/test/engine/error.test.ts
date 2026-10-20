import { expect, it } from 'vitest'
import { ContextWindowOverflowError } from '@strands-agents/sdk'
import { modelError } from '../../src/engine/error.js'
import { chunk, collect, setup } from '../../fixtures/engine.js'

it('preserves a wrapped worker overflow as the original cause', async () => {
  const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
  const error = new Error('ContextWindowSizeExceededError: input is too long.')
  create.mockRejectedValueOnce(error)
  const failed = collect(model.stream([]))
  await expect(failed).rejects.toBeInstanceOf(ContextWindowOverflowError)
  await expect(failed).rejects.toHaveProperty('cause', error)
  await expect(collect(model.stream([]))).resolves.toContainEqual({
    type: 'modelMessageStopEvent',
    stopReason: 'endTurn',
  })
})

it('keeps an existing Strands overflow error unchanged', () => {
  const error = new ContextWindowOverflowError('ContextWindowSizeExceededError: input is too long.')
  expect(modelError(error)).toBe(error)
})
