import { expect, it } from 'vitest'
import { gemmaPrompt } from '../../shared/web-llm/gemma/prompt.js'
import type { ProtocolMessage } from '../../shared/web-llm/types.js'

it('renders a frozen history repeatedly without consuming its system message', () => {
  const history: ProtocolMessage[] = [
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
  const history: ProtocolMessage[] = [
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

it('uses canonical mixed-case order for declarations, calls and structured results', () => {
  const values = { z: 1, B: 2, a: 3, A: 4, b: 5 }
  const tools = [
    {
      name: 'lookup',
      description: 'Read values.',
      inputSchema: {
        type: 'object' as const,
        properties: Object.fromEntries(
          Object.keys(values).map((key) => [key, { type: 'integer' as const }]),
        ),
      },
    },
  ]
  const history: ProtocolMessage[] = [
    { role: 'user', content: 'Read values.' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'call',
          type: 'function',
          function: { name: 'lookup', arguments: JSON.stringify(values) },
        },
      ],
    },
    { role: 'tool', content: '', tool_call_id: 'call', response: values },
  ]
  const before = structuredClone({ history, tools })
  const prompt = gemmaPrompt(history, tools)
  const fields = ['a', 'A', 'B', 'b', 'z'].map((key) => key + ':{type:<|"|>INTEGER<|"|>}').join(',')

  expect(prompt).toContain('properties:{' + fields + '}')
  expect(prompt).toContain('<|tool_call>call:lookup{a:3,A:4,B:2,b:5,z:1}<tool_call|>')
  expect(prompt).toContain('<|tool_response>response:lookup{a:3,A:4,B:2,b:5,z:1}<tool_response|>')
  expect({ history, tools }).toEqual(before)
})
