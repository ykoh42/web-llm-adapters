import { expect, it } from 'vitest'
import { qwenHistory } from '../../../../shared/web-llm/qwen/history.js'
import { canUseXmlFunctionName } from '../../../../shared/web-llm/qwen35/constraints.js'
import { formatCall, toolPrompt } from '../../../../shared/web-llm/qwen35/tools.js'

function tool(name: string) {
  return { name, description: 'Read a value.', inputSchema: { type: 'object' as const } }
}

it('rejects missing function names without throwing during validation', () => {
  expect(canUseXmlFunctionName(undefined)).toBe(false)
})

it.each(['', 'lookup>report', '<lookup'])(
  'rejects Qwen3.5 function name %j before request formatting',
  (name) => {
    expect(() => toolPrompt([tool(name)])).toThrow(
      'Qwen3.5 tool names must be non-empty and cannot contain < or >.',
    )
    expect(() => qwenHistory('Qwen3.5-4B-q4f16_1-MLC', [], [tool(name)], undefined, true)).toThrow(
      'Qwen3.5 tool names must be non-empty and cannot contain < or >.',
    )
  },
)

it.each(['', 'lookup>report', '<lookup'])(
  'rejects Qwen3.5 history function name %j before serialization',
  (name) => {
    expect(() => formatCall(name, {})).toThrow(
      'Qwen3.5 tool names must be non-empty and cannot contain < or >.',
    )
  },
)

it.each([
  { marker: '<|im_start|>', input: { text: 'before <|im_start|> after' } },
  { marker: '<|im_end|>', input: { text: 'before <|im_end|> after' } },
  { marker: '<|endoftext|>', input: { payload: { text: 'before <|endoftext|> after' } } },
])('rejects Qwen3.5 history values containing $marker', ({ input }) => {
  expect(() => formatCall('lookup', input)).toThrow('cannot preserve message-boundary tokens')
})

it.each(['<|im_start|>', '<|im_end|>', '<|endoftext|>'])(
  'rejects Qwen3.5 assistant history containing message-boundary token %s',
  (marker) => {
    const history = [
      {
        role: 'assistant' as const,
        content: '',
        tool_calls: [
          {
            id: 'call-1',
            type: 'function' as const,
            function: {
              name: 'lookup',
              arguments: JSON.stringify({ text: `before ${marker} after` }),
            },
          },
        ],
      },
    ]
    expect(() =>
      qwenHistory('Qwen3.5-4B-q4f16_1-MLC', history, [tool('lookup')], undefined, true),
    ).toThrow('cannot preserve message-boundary tokens')
  },
)
