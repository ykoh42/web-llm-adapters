import type { JSONSchema, ToolSpec } from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { chunk, chunks, collect } from '../../fixtures/engine.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'

it.each([undefined, { type: 'number' }, { type: 'object' }] as const)(
  'rejects overflowing tool numbers without replacing them with null: %j',
  async (schema) => {
    const value = schema?.type === 'object' ? '{"nested":[1e999]}' : '1e999'
    const content = `<tool_call><function=read><parameter=value>${value}</parameter></function></tool_call>`
    const tools: ToolSpec[] = [
      {
        name: 'read',
        description: '',
        ...(schema && { inputSchema: { type: 'object', properties: { value: schema } } }),
      },
    ]
    await expect(collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))).rejects.toThrow()
  },
)

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

it('reads a parameter closed by the next parameter, as the recommended parser does', async () => {
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=read>' +
              '<parameter=path>\n/tmp/note.txt\n' +
              '<parameter=limit>\n100\n</parameter>' +
              '</function></tool_call>',
          },
          'stop',
        ),
      ),
      [
        {
          name: 'read',
          description: 'Read a file.',
          inputSchema: {
            type: 'object',
            properties: { path: { type: 'string' }, limit: { type: 'number' } },
          },
        },
      ],
    ),
  )
  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { name: 'read', input: '{"path":"/tmp/note.txt","limit":100}' },
  ])
})

it.each(['</parameter>', '</parameter>\n</parameter>'])(
  'reads complete parameters with redundant closing tags: %s',
  async (closing) => {
    const tools: ToolSpec[] = [
      {
        name: 'read',
        description: 'Read a file.',
        inputSchema: {
          type: 'object',
          properties: { path: { type: 'string' }, limit: { type: 'number' } },
          required: ['path'],
        },
      },
    ]
    const output = await collect(
      qwen35Chunks(
        chunks(
          chunk(
            {
              content:
                '<tool_call>\n<function=read>\n' +
                '<parameter=path>\n/tmp/q35-40625.txt\n</parameter>\n' +
                '<parameter=limit>\n100\n</parameter>\n' +
                `${closing}\n</function>\n</tool_call>`,
            },
            'stop',
          ),
        ),
        tools,
      ),
    )

    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { name: 'read', input: '{"path":"/tmp/q35-40625.txt","limit":100}' },
    ])
  },
)

it.each(['stray text', '<parameter=limit>100', '</parameter>stray text'])(
  'rejects incomplete or unknown parameter content: %s',
  async (extra) => {
    const output = qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=city>Seoul</parameter>' +
              `${extra}</function></tool_call>`,
          },
          'stop',
        ),
      ),
      [{ name: 'lookup', description: 'Look up a city.' }],
    )

    await expect(collect(output)).rejects.toThrow('malformed tool parameter')
  },
)

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

it.each([
  ['length', 'maxTokens'],
  ['abort', 'cancelled'],
] as const)('returns a truncated tool call as text on %s', async (finishReason, stop) => {
  const raw = '<tool_call><function=lookup><parameter=city>Seoul'
  const tools: ToolSpec[] = [{ name: 'lookup', description: 'Look up a city.' }]
  const output = await collect(qwen35Chunks(chunks(chunk({ content: raw }, finishReason)), tools))

  expect(output.map((part) => part.text).join('')).toBe(raw)
  expect(output.at(-1)?.stop).toBe(stop)
  expect(output.flatMap((item) => item.calls)).toEqual([])
})

it.each([1, 7, 128])('returns earlier complete calls across %i-character chunks', async (size) => {
  const complete =
    '<tool_call><function=lookup><parameter=city>Seoul</parameter></function></tool_call>'
  const partial = '<tool_call><function=lookup><parameter=city>Tokyo'
  const raw = complete + partial
  const parts = raw.match(new RegExp(`[\\s\\S]{1,${size}}`, 'g'))!
  const output = await collect(
    qwen35Chunks(
      chunks(
        ...parts.map((content, index) =>
          chunk({ content }, index === parts.length - 1 ? 'length' : null),
        ),
      ),
      [{ name: 'lookup', description: 'Look up a city.' }],
    ),
  )

  expect(output.map((part) => part.text).join('')).toBe(raw)
  expect(output.flatMap((part) => part.calls)).toHaveLength(1)
  expect(output.at(-1)?.stop).toBe('maxTokens')
})
