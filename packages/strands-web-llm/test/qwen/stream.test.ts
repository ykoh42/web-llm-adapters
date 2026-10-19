import type { ToolSpec } from '@strands-agents/sdk'
import { expect, it, vi } from 'vitest'
import { chunk, chunks, collect } from '../../fixtures/engine.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'

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
    collect(qwenChunks(chunks(raw), [{ name: 'read', description: 'Read a file.' }], onChunk)),
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

it('parses a complete tool call when the model omits the closing tag', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(
    qwenChunks(
      chunks(
        chunk({ content: '<tool_call>{"name": "lookup", "arguments": {"city": "Seoul"}}' }, 'stop'),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"Seoul"}' },
  ])
})

it('parses a complete tool call with one extra closing brace', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(
    qwenChunks(
      chunks(
        chunk(
          {
            content: '<tool_call>{"name":"lookup","arguments":{"city":"Seoul"}}}</tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"Seoul"}' },
  ])
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
