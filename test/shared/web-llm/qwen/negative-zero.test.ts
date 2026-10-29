import { expect, it } from 'vitest'
import { chunk, chunks, collect } from '../../../../fixtures/web-llm.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'
import { toolJSON } from '../../../../shared/web-llm/tool-json.js'

const tools = [
  {
    name: 'lookup',
    description: '',
    inputSchema: {
      type: 'object' as const,
      properties: { value: { type: 'object' as const } },
    },
  },
]
const value = '{"__proto__":{"values":[-0.0,0,-1e-9999]},"text":"-0"}'

it.each([
  {
    name: 'Qwen3',
    parse: qwenChunks,
    content: `<tool_call>{"name":"lookup","arguments":{"value":${value}}}</tool_call>`,
  },
  {
    name: 'Qwen3.5',
    parse: qwen35Chunks,
    content: `<tool_call><function=lookup><parameter=value>${value}</parameter></function></tool_call>`,
  },
])('$name keeps negative zero in nested tool arguments', async ({ parse, content }) => {
  const output = await collect(parse(chunks(chunk({ content }, 'stop')), tools))
  const args = JSON.parse(output.flatMap((part) => part.calls)[0]!.input!)
  const nested = Object.getOwnPropertyDescriptor(args.value, '__proto__')!.value
  expect(Object.is(nested.values[0], -0)).toBe(true)
  expect(Object.is(nested.values[1], 0)).toBe(true)
  expect(Object.is(nested.values[2], -0)).toBe(true)
  expect(args.value.text).toBe('-0')
})

it('serializes deeply nested values without recursion', () => {
  const source = '['.repeat(3000) + '-0.0' + ']'.repeat(3000)
  const value: unknown = JSON.parse(source)
  expect(toolJSON(value, source)).toBe('['.repeat(3000) + '-0' + ']'.repeat(3000))
})

it('preserves other JSON values alongside negative zero', () => {
  const value = {
    number: -0,
    text: 'A "quote", a \\ path, a newline\n, and 서울',
    values: [null, true, false, 1e-300, -1e300, { nested: [] }],
  }
  expect(JSON.parse(toolJSON(value, '-0'))).toStrictEqual(value)
})
