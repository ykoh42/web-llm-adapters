import type { StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'
import type { ToolSpec } from '../../../../shared/web-llm/types.js'
import { expect, it, vi } from 'vitest'
import { chunk, chunks, collect } from '../../../../fixtures/web-llm.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { formatCall } from '../../../../shared/web-llm/qwen/tools.js'

it.each(['1e999', '{"nested":[1e999]}'])(
  'rejects overflowing tool numbers instead of changing them to null: %s',
  async (value) => {
    const content = `<tool_call>{"name":"read","arguments":{"value":${value}}}</tool_call>`
    await expect(
      collect(qwenChunks(chunks(chunk({ content }, 'stop')), [{ name: 'read', description: '' }])),
    ).rejects.toThrow()
  },
)

it.each([1, 7, 128])(
  'keeps literal tool tags and escaped quotes in %i-character chunks',
  async (size) => {
    const input = { content: 'A quoted "</tool_call>" tag, a \\ path, and <tool_call> text.' }
    const source = formatCall('write_file', input)
    const parts = source.match(new RegExp(`[\\s\\S]{1,${size}}`, 'g'))!
    const output = await collect(
      qwenChunks(
        chunks(
          ...parts.map((content, index) =>
            chunk({ content }, index === parts.length - 1 ? 'stop' : null),
          ),
        ),
        [{ name: 'write_file', description: 'Write text.' }],
      ),
    )
    expect(output.flatMap((part) => part.calls)).toMatchObject([
      { name: 'write_file', input: JSON.stringify(input) },
    ])
  },
)

it('rejects an unregistered function wrapper without changing the model output', async () => {
  const raw = chunk(
    {
      content:
        '<tool_call>\n{"name":"function","function":{"name":"read"},"arguments":{"path":"/tmp/note.txt"}}\n</tool_call>',
    },
    'stop',
  )
  const onChunk = vi.fn()

  await expect(
    collect(qwenChunks(chunks(raw), [{ name: 'read', description: 'Read a file.' }], { onChunk })),
  ).rejects.toThrow('Qwen returned unknown tool "function".')
  expect(onChunk).toHaveBeenCalledExactlyOnceWith(raw)
})

it('splits thinking and parses a tool call across partial tags', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(
    qwenChunks(
      chunks(
        chunk({ content: '<thi' }),
        chunk({
          content: 'nk>Plan.</think><tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}',
        }),
        chunk({ content: '</tool_call>' }, 'stop'),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"Seoul"}' },
  ])
  expect(output.map((item) => item.reasoning).join('')).toBe('Plan.')
  expect(output.map((item) => item.text).join('')).toBe('')
})

it.each([
  {
    name: 'missing closing tag',
    content: '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}',
  },
  {
    name: 'extra JSON delimiter',
    content: '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}}</tool_call>',
  },
])('rejects a malformed tool call with $name and keeps its raw chunk', async ({ content }) => {
  const raw = chunk({ content }, 'stop')
  const onChunk = vi.fn()
  const output: StreamChunk[] = []

  await expect(async () => {
    for await (const value of qwenChunks(
      chunks(raw),
      [{ name: 'lookup', description: 'Look up a city.' }],
      { onChunk },
    ))
      output.push(value)
  }).rejects.toThrow()

  expect(output.flatMap((item) => item.calls)).toEqual([])
  expect(output.map((item) => item.text).join('')).toBe('')
  expect(onChunk).toHaveBeenCalledExactlyOnceWith(raw)
})

it.each([
  ['length', 'maxTokens'],
  ['abort', 'cancelled'],
] as const)(
  'returns an incomplete tool call without exposing it as text on %s',
  async (finishReason, stop) => {
    const raw = '<tool_call>{"name":"lookup","arguments":{"city":"Se'
    const output = await collect(
      qwenChunks(chunks(chunk({ content: raw }, finishReason)), [
        { name: 'lookup', description: 'Look up a city.' },
      ]),
    )

    expect(output.map((part) => part.text).join('')).toBe('')
    expect(output.flatMap((part) => part.calls)).toEqual([])
    expect(output.at(-1)?.stop).toBe(stop)
  },
)

it.each([1, 7, 128])('returns earlier complete calls across %i-character chunks', async (size) => {
  const complete = '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}</tool_call>'
  const partial = '<tool_call>{"name":"lookup","arguments":{"city":"To'
  const raw = complete + partial
  const parts = raw.match(new RegExp(`[\\s\\S]{1,${size}}`, 'g'))!
  const output = await collect(
    qwenChunks(
      chunks(
        ...parts.map((content, index) =>
          chunk({ content }, index === parts.length - 1 ? 'length' : null),
        ),
      ),
      [{ name: 'lookup', description: 'Look up a city.' }],
    ),
  )

  expect(output.map((part) => part.text).join('')).toBe('')
  expect(output.flatMap((part) => part.calls)).toHaveLength(1)
  expect(output.at(-1)?.stop).toBe('maxTokens')
})

it('keeps distinct calls to the same tool', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(
    qwenChunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}</tool_call>' +
              '<tool_call>{"name":"lookup","arguments":{"city":"Tokyo"}}</tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"Seoul"}' },
    { index: 1, name: 'lookup', input: '{"city":"Tokyo"}' },
  ])
})

it('preserves repeated identical calls from the model', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const call = '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}</tool_call>'
  const output = await collect(qwenChunks(chunks(chunk({ content: call + call }, 'stop')), tools))

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"Seoul"}' },
    { index: 1, name: 'lookup', input: '{"city":"Seoul"}' },
  ])
})
