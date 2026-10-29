import { expect, it } from 'vitest'
import { chunk, chunks } from '../../../fixtures/web-llm.js'
import { nativeChunks } from '../../../shared/web-llm/native.js'

it.each([
  { first: ' note', rest: ']' },
  { first: ']', rest: '' },
  { first: '[1]', rest: ']' },
])(
  'streams bracketed text once it cannot be a native tool call: $first',
  async ({ first, rest }) => {
    const stream = nativeChunks(
      chunks(chunk({ content: '[' }), chunk({ content: first }), chunk({ content: rest }, 'stop')),
      true,
    )
    const iterator = stream[Symbol.asyncIterator]()
    expect((await iterator.next()).value?.text).toBe('')
    expect((await iterator.next()).value?.text).toBe('[' + first)
    expect((await iterator.next()).value?.text).toBe(rest)
    expect((await iterator.next()).done).toBe(true)
  },
)

it('holds split tool JSON until it can compare the native call', async () => {
  const stream = nativeChunks(
    chunks(
      chunk({ content: '[' }),
      chunk({ content: '  ' }),
      chunk({ content: '{"name":"lookup","arguments":{}}]' }),
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'lookup', arguments: '{}' } }] },
        'tool_calls',
      ),
    ),
    true,
  )
  const output = []
  for await (const item of stream) output.push(item)

  expect(output.map((item) => item.text)).toEqual(['', '', '', ''])
  expect(output.at(-1)?.calls).toMatchObject([{ name: 'lookup', input: '{}' }])
})
