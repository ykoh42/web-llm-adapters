import { expect, it } from 'vitest'
import { gemmaPrompt, type GemmaMessage } from '../../shared/web-llm/gemma/prompt.js'

it('renders a frozen history repeatedly without consuming its system message', () => {
  const history: GemmaMessage[] = [
    { role: 'system', content: '  Caller  ' },
    { role: 'user', content: '  Hello  ' },
    { role: 'assistant', content: 'First', reasoning: 'Plan' },
    { role: 'assistant', content: 'Second' },
    { role: 'user', content: '  Next  ' },
  ]
  const before = structuredClone(history)
  history.forEach(Object.freeze)
  Object.freeze(history)
  const expected =
    '<|turn>system\n<|think|>\nCaller<turn|>\n' +
    '<|turn>user\nHello<turn|>\n' +
    '<|turn>model\n<|channel>thought\nPlan\n<channel|>FirstSecond<turn|>\n' +
    '<|turn>user\nNext<turn|>\n<|turn>model\n'
  expect(gemmaPrompt(history, [], true)).toBe(expected)
  expect(gemmaPrompt(history, [], true)).toBe(expected)
  expect(history).toEqual(before)
})

it('renders tool response context without altering the original messages', () => {
  const history: GemmaMessage[] = [
    { role: 'system', content: '' },
    { role: 'user', content: 'Look up A7.' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'call', type: 'function', function: { name: 'lookup', arguments: '{"code":"A7"}' } },
      ],
    },
    {
      role: 'tool',
      tool_call_id: 'call',
      content: '42',
      response: { answer: 42 },
      context: 'Next',
    },
  ]
  const before = structuredClone(history)
  Object.freeze(history)
  const prompt = gemmaPrompt(history, [])
  expect(prompt).toContain('response:lookup{context:<|"|>Next<|"|>,value:{answer:42}}')
  expect(gemmaPrompt(history, [])).toBe(prompt)
  expect(history).toEqual(before)
})
