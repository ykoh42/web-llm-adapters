import type { JSONSchema, ToolSpec } from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { chunk, chunks, collect } from '../../fixtures/engine.js'
import { qwen35Chunks } from '../../src/qwen35/stream.js'

const nullableStringSchemas: JSONSchema[] = [
  { type: ['string', 'null'] },
  { anyOf: [{ type: 'string' }, { type: 'null' }] },
  { $ref: '#/definitions/city' },
]

const stringRootSchemas: JSONSchema[] = [
  {
    $ref: '#/definitions/input',
    definitions: {
      input: { type: 'object', properties: { city: { type: 'string' } } },
    },
  },
  {
    allOf: [
      { type: 'object', properties: { city: { type: 'string' } } },
      { description: 'Shared tool input.' },
    ],
  },
  { type: 'object', additionalProperties: { type: 'string' } },
]

const narrowedStringSchemas: JSONSchema[] = [
  { type: 'string', anyOf: [{ type: 'string' }, { type: 'null' }] },
  { type: ['string', 'null'], allOf: [{ type: 'string' }] },
  { $ref: '#/definitions/city', type: 'string' },
]

it.each(nullableStringSchemas)('keeps a nullable string argument as text', async (city) => {
  const inputSchema: JSONSchema = {
    type: 'object',
    properties: { city },
    definitions: { city: { type: ['string', 'null'] } },
  }
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.', inputSchema }]
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=city>123</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"123"}' },
  ])
})

it.each(stringRootSchemas)(
  'keeps a string argument as text in root schema shapes',
  async (inputSchema) => {
    const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.', inputSchema }]
    const output = await collect(
      qwen35Chunks(
        chunks(
          chunk(
            {
              content:
                '<tool_call><function=lookup><parameter=city>123</parameter></function></tool_call>',
            },
            'stop',
          ),
        ),
        tools,
      ),
    )

    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { name: 'lookup', input: '{"city":"123"}' },
    ])
  },
)

it('intersects overlapping property types in allOf schemas', async () => {
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a city.',
      inputSchema: {
        allOf: [
          { properties: { city: { type: ['string', 'null'] } } },
          { properties: { city: { type: 'string' } } },
        ],
      },
    },
  ]
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=city>null</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { name: 'lookup', input: '{"city":"null"}' },
  ])
})

it.each(narrowedStringSchemas)('applies combined property schema constraints', async (city) => {
  const inputSchema: JSONSchema = {
    type: 'object',
    properties: { city },
    definitions: { city: { type: ['string', 'null'] } },
  }
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.', inputSchema }]
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=city>null</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { name: 'lookup', input: '{"city":"null"}' },
  ])
})

it('keeps null as null for a nullable string argument', async () => {
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a city.',
      inputSchema: { type: 'object', properties: { city: { type: ['string', 'null'] } } },
    },
  ]
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=city>null</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { name: 'lookup', input: '{"city":null}' },
  ])
})

it('rejects a schema union when the XML value could be text or a number', async () => {
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a city.',
      inputSchema: {
        type: 'object',
        properties: { city: { anyOf: [{ type: 'string' }, { type: 'number' }] } },
      },
    },
  ]

  await expect(
    collect(
      qwen35Chunks(
        chunks(
          chunk(
            {
              content:
                '<tool_call><function=lookup><parameter=city>123</parameter></function></tool_call>',
            },
            'stop',
          ),
        ),
        tools,
      ),
    ),
  ).rejects.toThrow('ambiguous tool argument')
})

it('preserves an argument named __proto__', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=__proto__>{"city":"Seoul"}</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { input: '{"__proto__":{"city":"Seoul"}}' },
  ])
})

it('rejects an unclosed Qwen3.5 parameter instead of dropping it', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]

  await expect(
    collect(
      qwen35Chunks(
        chunks(
          chunk(
            {
              content: '<tool_call><function=lookup><parameter=city>Seoul</function></tool_call>',
            },
            'stop',
          ),
        ),
        tools,
      ),
    ),
  ).rejects.toThrow('malformed tool parameter')
})

it('preserves the token limit when a tool call is truncated', async () => {
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(
    qwen35Chunks(
      chunks(chunk({ content: '<tool_call><function=lookup><parameter=city>Seoul' }, 'length')),
      tools,
    ),
  )

  expect(output.at(-1)?.stop).toBe('maxTokens')
  expect(output.flatMap((item) => item.calls)).toEqual([])
})
