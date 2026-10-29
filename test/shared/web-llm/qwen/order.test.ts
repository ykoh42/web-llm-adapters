import { expect, it } from 'vitest'
import { chunk, chunks, collect } from '../../../../fixtures/web-llm.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { formatCall as formatQwenCall } from '../../../../shared/web-llm/qwen/tools.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'
import { formatCall as formatQwen35Call } from '../../../../shared/web-llm/qwen35/tools.js'

const tools = [
  {
    name: 'lookup',
    description: 'Look up a code.',
    inputSchema: { type: 'object' as const, properties: { code: { type: 'string' as const } } },
  },
]

it.each([
  { name: 'Qwen', parse: qwenChunks, format: formatQwenCall },
  { name: 'Qwen3.5', parse: qwen35Chunks, format: formatQwen35Call },
])('keeps $name output order across every two-chunk split', async ({ parse, format }) => {
  const text =
    'Before.<think>Plan.</think>During.' +
    format('lookup', { code: 'A7' }) +
    '<think>Review.</think>Between.' +
    format('lookup', { code: 'B8' }) +
    'After.'
  for (let split = 0; split <= text.length; split++) {
    const output = await collect(
      parse(
        chunks(
          chunk({ content: text.slice(0, split) }),
          chunk({ content: text.slice(split) }, 'stop'),
        ),
        tools,
      ),
    )
    const parts: string[] = []
    for (const part of output) {
      if (part.text) parts.push(`text:${part.text}`)
      if (part.reasoning) parts.push(`thinking:${part.reasoning}`)
      for (const call of part.calls) parts.push(`tool:${call.name}:${call.input}`)
    }
    // Adjacent text pieces may be emitted separately at a chunk boundary.
    const combined = parts.reduce<string[]>((result, part) => {
      const at = part.indexOf(':')
      const previous = result.at(-1)
      if (previous && !part.startsWith('tool:') && previous.startsWith(part.slice(0, at + 1)))
        result[result.length - 1] += part.slice(at + 1)
      else result.push(part)
      return result
    }, [])
    expect(combined, `split at ${split}`).toEqual([
      'text:Before.',
      'thinking:Plan.',
      'text:During.',
      'tool:lookup:{"code":"A7"}',
      'thinking:Review.',
      'text:Between.',
      'tool:lookup:{"code":"B8"}',
      'text:After.',
    ])
  }
})
