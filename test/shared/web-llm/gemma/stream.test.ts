import type { Completion } from '@mlc-ai/web-llm'
import { expect, it } from 'vitest'
import { gemmaChunks } from '../../../../shared/web-llm/gemma/stream.js'
import type { ToolSpec } from '../../../../shared/web-llm/types.js'

it.each([
  { name: 'lookup', separator: ' ', label: 'a space' },
  { name: 'lookup', separator: '\n\t', label: 'newlines and tabs' },
  { name: 'read <file>', separator: '', label: 'a name containing spaces' },
  {
    name: 'lookup user',
    separator: '',
    label: 'the longest registered name sharing a prefix',
    registeredNames: ['lookup', 'lookup user'],
  },
])(
  'parses $name with $label before its arguments',
  async ({ name, separator, registeredNames }) => {
    const text = `<|tool_call>call:${name}${separator}{code:<|"|>A7<|"|>}<tool_call|>`
    const completion: Completion = {
      id: 'test',
      model: 'test-model',
      object: 'text_completion',
      created: 0,
      choices: [{ index: 0, text, finish_reason: 'stop' }],
    }
    const tools: ToolSpec[] = (registeredNames ?? [name]).map((name) => ({
      name,
      description: 'Look up a code.',
    }))
    const chunks = (async function* () {
      yield completion
    })()
    const output = []

    for await (const chunk of gemmaChunks(chunks, tools)) output.push(chunk)

    expect(output.flatMap((chunk) => chunk.calls)).toMatchObject([
      { name, input: JSON.stringify({ code: 'A7' }) },
    ])
  },
)
