import { expect, it } from 'vitest'
import { checkThinkingPrefill } from '../../../../shared/web-llm/qwen/prefill.js'

it.each([undefined, false])('allows prefill=%s for any model', (prefilled) => {
  expect(() => checkThinkingPrefill('Gemma-test', prefilled)).not.toThrow()
})

it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  'allows thinking prefill for %s',
  (modelId) => {
    expect(() => checkThinkingPrefill(modelId, true)).not.toThrow()
  },
)

it.each(['test-model', 'Qwen2.5-3B-Instruct-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC'])(
  'rejects thinking prefill for %s',
  (modelId) => {
    expect(() => checkThinkingPrefill(modelId, true)).toThrow(
      'thinkingPrefilled requires a Qwen thinking model.',
    )
  },
)

it.each([null, 1, 'true'])('rejects a non-boolean prefill value: %s', (prefilled) => {
  expect(() => checkThinkingPrefill('Qwen3-4B-q4f16_1-MLC', prefilled as never)).toThrow(
    new TypeError('thinkingPrefilled must be a boolean.'),
  )
})
