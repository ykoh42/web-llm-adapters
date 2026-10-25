import { expect, it } from 'vitest'
import { formatQwenAssistant } from '../../shared/web-llm/qwen/history.js'

it('preserves internal newlines in a long current-turn thought', () => {
  const thought = 'Plan' + '\n'.repeat(40_000) + 'Next'
  expect(formatQwenAssistant('\nAnswer\n', '\n\n' + thought + '\n\n', ['Call'])).toBe(
    '<think>\n' + thought + '\n</think>\n\nAnswer\n\nCall',
  )
})

it('removes only line feeds from Qwen JSON thought edges', () => {
  const thought = ' \tPlan\r\u2028\u2029\u00a0'
  expect(formatQwenAssistant('\n Answer \n', '\n' + thought + '\n', ['First', 'Second'])).toBe(
    '<think>\n' + thought + '\n</think>\n\n Answer \n\nFirst\nSecond',
  )
})

it.each(['', '\n', '\n\n\n'])('does not add thinking for %j', (thought) => {
  expect(formatQwenAssistant('\nAnswer\n', thought, ['Call'])).toBe('\nAnswer\n\nCall')
})

it('keeps the separate whitespace and tool separators used by Qwen XML', () => {
  expect(formatQwenAssistant('\n Answer \n', '\n \tPlan\r\n', ['First', 'Second'], true)).toBe(
    '<think>\nPlan\n</think>\n\nAnswer\n\nFirst\nSecond',
  )
})

it.each(['', '\n\n'])('keeps the current-turn Qwen XML thinking boundary for %j', (thought) => {
  expect(formatQwenAssistant('', thought, ['Call'], true)).toBe('<think>\n\n</think>\n\nCall')
})

it('omits the thinking boundary for earlier Qwen XML turns', () => {
  expect(formatQwenAssistant('Old answer', undefined, ['Call'], true)).toBe('Old answer\n\nCall')
})
