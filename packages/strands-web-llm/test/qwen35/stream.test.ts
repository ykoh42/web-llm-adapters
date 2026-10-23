import type { JSONSchema, ToolSpec } from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { chunk, chunks, collect } from '../../fixtures/engine.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'
import { formatCall } from '../../../../shared/web-llm/qwen35/tools.js'

it.each(['code', '', 'a<b', 'x\ny', '__proto__', 'constructor'])(
  'preserves a representable XML argument name: %j',
  async (name) => {
    const input = { [name]: 42 }
    const content = formatCall('lookup', input)
    const tools: ToolSpec[] = [
      {
        name: 'lookup',
        description: 'Read a value.',
        inputSchema: { type: 'object', properties: { [name]: { type: 'number' } } },
      },
    ]
    const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
    expect(output.flatMap((item) => item.calls)).toMatchObject([{ input: JSON.stringify(input) }])
  },
)

it('preserves angle brackets in nested JSON keys and ordinary string values', async () => {
  const input = { payload: { 'a>b': 42 }, text: 'a > b' }
  const content = formatCall('lookup', input)
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Read a value.',
      inputSchema: {
        type: 'object',
        properties: { payload: { type: 'object' }, text: { type: 'string' } },
      },
    },
  ]
  const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
  expect(output.flatMap((item) => item.calls)).toMatchObject([{ input: JSON.stringify(input) }])
})

it('analyzes each argument schema once per stream', async () => {
  let reads = 0
  const payload = {
    get type() {
      reads++
      return 'object' as const
    },
  }
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Read a value.',
      inputSchema: { type: 'object', properties: { payload } },
    },
  ]
  const content = formatCall('lookup', { payload: { value: 42 } })
  const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))

  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { input: JSON.stringify({ payload: { value: 42 } }) },
  ])
  expect(reads).toBe(1)
})

it.each([
  { value: { text: 'literal <parameter=x> inside JSON' }, type: 'object' },
  { value: { text: 'literal </parameter> </function> </tool_call> inside JSON' }, type: 'object' },
  { value: ['<parameter=x>', '</parameter>', '</function>', '</tool_call>'], type: 'array' },
] satisfies { value: unknown; type: 'object' | 'array' }[])(
  'preserves XML tags inside a JSON $type argument',
  async ({ value, type }) => {
    const tools: ToolSpec[] = [
      {
        name: 'lookup',
        description: 'Read a value.',
        inputSchema: { type: 'object', properties: { payload: { type } } },
      },
    ]
    const content = formatCall('lookup', { payload: value })
    const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { name: 'lookup', input: JSON.stringify({ payload: value }) },
    ])
  },
)

it.each(['{not JSON " remains', '[raw "value', 'literal "unterminated'])(
  'preserves an ordinary string that resembles incomplete JSON: %s',
  async (text) => {
    const tools: ToolSpec[] = [
      {
        name: 'lookup',
        description: 'Read text.',
        inputSchema: { type: 'object', properties: { text: { type: 'string' } } },
      },
    ]
    const content = formatCall('lookup', { text })
    const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { input: JSON.stringify({ text }) },
    ])
  },
)

it.each([
  { $ref: '#/definitions/payload' },
  { anyOf: [{ type: 'object' }, { type: 'null' }] },
  { allOf: [{ type: 'object' }, { description: 'Structured data.' }] },
  { const: { text: '<parameter=x> </parameter> </tool_call>' } },
] satisfies JSONSchema[])(
  'preserves XML in structured arguments inferred from %j',
  async (schema) => {
    const payload = { text: '<parameter=x> </parameter> </tool_call>' }
    const content = formatCall('lookup', { payload })
    const output = await collect(
      qwen35Chunks(chunks(chunk({ content }, 'stop')), [
        {
          name: 'lookup',
          description: 'Read data.',
          inputSchema: {
            type: 'object',
            properties: { payload: schema },
            definitions: { payload: { type: 'object' } },
          },
        },
      ]),
    )
    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { input: JSON.stringify({ payload }) },
    ])
  },
)

it('reads several functions with whitespace and redundant parameter closers', async () => {
  const content =
    '<tool_call>\n<function=lookup>\n</parameter>\n' +
    '<parameter=city>\nSeoul\n</parameter>\n</parameter>\n</function>\n\n' +
    '<function=lookup>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>'
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a city.',
      inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
    },
  ]
  const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { index: 0, name: 'lookup', input: '{"city":"Seoul"}' },
    { index: 1, name: 'lookup', input: '{"city":"Tokyo"}' },
  ])
})

it.each([
  { source: '<function=lookup>\n'.repeat(8000), error: 'ended inside a tool call' },
  {
    source: '<function=lookup>' + '<parameter=city'.repeat(8000) + '</function>',
    error: 'malformed tool parameter',
  },
])('rejects repeated unclosed tags: $error', async ({ source, error }) => {
  const content = '<tool_call>' + source + '</tool_call>'
  await expect(
    collect(
      qwen35Chunks(chunks(chunk({ content }, 'stop')), [
        { name: 'lookup', description: 'Look up a city.' },
      ]),
    ),
  ).rejects.toThrow(error)
})

it('reads current schemas when tool definitions are reused across streams', async () => {
  const code: JSONSchema = { type: 'string' }
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a code.',
      inputSchema: { type: 'object', properties: { code } },
    },
  ]
  const content =
    '<tool_call><function=lookup><parameter=code>123</parameter></function></tool_call>'
  for (const type of ['string', 'number'] as const) {
    code.type = type
    const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { input: type === 'string' ? '{"code":"123"}' : '{"code":123}' },
    ])
  }
})

it.each([
  { enum: ['123', 'false', 'null'] },
  { const: '123' },
  { type: ['string', 'integer'], enum: ['123'] },
  { anyOf: [{ enum: ['123'] }, { type: 'null' }] },
  { allOf: [{ type: ['string', 'integer'] }, { const: '123' }] },
  { $ref: '#/definitions/code' },
] satisfies JSONSchema[])('keeps string values from enum and const schemas: %j', async (schema) => {
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a code.',
      inputSchema: {
        type: 'object',
        properties: { code: schema },
        definitions: { code: { enum: ['123'] } },
      },
    },
  ]
  const output = await collect(
    qwen35Chunks(
      chunks(
        chunk(
          {
            content:
              '<tool_call><function=lookup><parameter=code>123</parameter></function></tool_call>',
          },
          'stop',
        ),
      ),
      tools,
    ),
  )
  expect(output.flatMap((item) => item.calls)).toMatchObject([
    { name: 'lookup', input: '{"code":"123"}' },
  ])
})

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

it.each(['boolean', 'object', 'array'] as const)(
  'keeps numeric-looking strings when the other type is %s',
  async (type) => {
    const content =
      '<tool_call><function=read><parameter=value>1e999</parameter></function></tool_call>'
    const tools: ToolSpec[] = [
      {
        name: 'read',
        description: '',
        inputSchema: {
          type: 'object',
          properties: { value: { anyOf: [{ type: 'string' }, { type }] } },
        },
      },
    ]
    const output = await collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))
    expect(output.flatMap((item) => item.calls)).toMatchObject([
      { name: 'read', input: '{"value":"1e999"}' },
    ])
  },
)

it.each([
  { type: 'number', value: '1e999' },
  { type: 'integer', value: '1e999' },
  { type: 'object', value: '{"nested":1e999}' },
  { type: 'array', value: '[1e999]' },
] as const)('rejects overflowing ambiguous string or $type arguments', async ({ type, value }) => {
  const content = `<tool_call><function=read><parameter=value>${value}</parameter></function></tool_call>`
  const tools: ToolSpec[] = [
    {
      name: 'read',
      description: '',
      inputSchema: {
        type: 'object',
        properties: { value: { anyOf: [{ type: 'string' }, { type }] } },
      },
    },
  ]
  await expect(collect(qwen35Chunks(chunks(chunk({ content }, 'stop')), tools))).rejects.toThrow()
})

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

it.each([
  {
    content:
      '<tool_call><function=lookup>unexpected<parameter=city>Seoul</parameter></function></tool_call>',
    error: 'malformed tool parameter',
  },
  {
    content:
      '<tool_call><function=lookup><parameter=city>Seoul</parameter>' +
      '<parameter=city>Tokyo</parameter></function></tool_call>',
    error: 'repeated parameter "city"',
  },
])('rejects malformed Qwen3.5 tool arguments: $error', async ({ content, error }) => {
  await expect(
    collect(
      qwen35Chunks(chunks(chunk({ content }, 'stop')), [
        { name: 'lookup', description: 'Look up a city.' },
      ]),
    ),
  ).rejects.toThrow(error)
})

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

it.each([
  { anyOf: [{ type: 'string' }, { type: 'number' }] },
  { enum: ['123', 123] },
] satisfies JSONSchema[])('rejects an ambiguous string or number argument: %j', async (city) => {
  const tools: ToolSpec[] = [
    {
      name: 'lookup',
      description: 'Look up a city.',
      inputSchema: {
        type: 'object',
        properties: { city },
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
