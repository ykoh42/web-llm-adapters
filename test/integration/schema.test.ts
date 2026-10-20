import { expect, it } from 'vitest'
import { Message, TextBlock, ToolUseBlock } from '@strands-agents/sdk'
import type { JSONSchema7 } from 'json-schema'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const messages = [new Message({ role: 'user', content: [new TextBlock('Find code 123.')] })]
const cases: { name: string; schema: JSONSchema7; text?: string; expected?: unknown }[] = [
  {
    name: 'encoded pointer escapes',
    schema: {
      definitions: { 'code/text': { type: 'string' } },
      properties: { code: { $ref: '#/definitions/code%7E1text' } },
    },
  },
  {
    name: 'encoded path separators',
    schema: {
      definitions: {
        input: { properties: { code: { type: 'string' } } },
        'input/properties/code': { type: 'number' },
      },
      properties: { code: { $ref: '#/definitions/input%2Fproperties%2Fcode' } },
    },
  },
  {
    name: 'encoded leading slash',
    schema: {
      definitions: { code: { type: 'string' } },
      properties: { code: { $ref: '#%2Fdefinitions%2Fcode' } },
    },
  },
  {
    name: 'array element',
    schema: {
      definitions: { code: { anyOf: [{ type: 'string' }] } },
      properties: { code: { $ref: '#/definitions/code/anyOf/0' } },
    },
  },
  {
    name: 'root document',
    schema: { type: 'object', properties: { code: { $ref: '#' } } },
    text: '{"nested":123}',
    expected: { nested: 123 },
  },
  {
    name: 'empty object key',
    schema: {
      definitions: { '': { type: 'string' } },
      properties: { code: { $ref: '#/definitions/' } },
    },
  },
  {
    name: 'literal pointer escape',
    schema: {
      definitions: { 'code~1text': { type: 'string' } },
      properties: { code: { $ref: '#/definitions/code~01text' } },
    },
  },
  {
    name: 'literal percent escape',
    schema: {
      definitions: { 'code%20text': { type: 'string' } },
      properties: { code: { $ref: '#/definitions/code%2520text' } },
    },
  },
  {
    name: 'own reserved key',
    schema: {
      definitions: { ['__proto__']: { type: 'string' } },
      properties: { code: { $ref: '#/definitions/__proto__' } },
    },
  },
]

async function read(client: string, schema: JSONSchema7, text = '123', name = 'code') {
  const { engine } = setup(
    chunk(
      {
        content:
          '<tool_call><function=lookup><parameter=' +
          name +
          '>' +
          text +
          '</parameter></function></tool_call>',
      },
      'stop',
    ),
  )
  if (client === 'strands') {
    const model = new WebLLMModel({ engine, modelId })
    const calls = []
    for await (const block of model.streamAggregated(messages, {
      toolSpecs: [{ name: 'lookup', description: 'Look up a code.', inputSchema: schema }],
    }))
      if (block instanceof ToolUseBlock) calls.push(block.input)
    return calls
  }
  const model = new ChatWebLLM({ engine, modelId })
  const result = await model
    .bindTools([
      {
        type: 'function',
        function: { name: 'lookup', description: 'Look up a code.', parameters: schema },
      },
    ])
    .invoke('Find code 123.')
  return result.tool_calls?.map((call) => call.args)
}

for (const client of ['strands', 'langchain']) {
  it.each(['access code', '', '""', ' code ', 'key\tname', 'line\nbreak', '🗝️ code'])(
    `${client} keeps the exact parameter name %j`,
    async (name) => {
      const schema: JSONSchema7 = {
        type: 'object',
        properties: { [name]: { type: 'string' } },
        required: [name],
      }
      expect(await read(client, schema, '123', name)).toEqual([{ [name]: '123' }])
    },
  )

  for (const keyword of ['allOf', 'anyOf'] as const) {
    it.each(['root', 'property'])(
      `${client} reads a shared ${keyword} schema at the %s`,
      async (scope) => {
        const definitions: Record<string, JSONSchema7> = {
          level0:
            scope === 'root' ? { properties: { code: { type: 'string' } } } : { type: 'string' },
        }
        for (let level = 1; level <= 17; level++) {
          const $ref = '#/definitions/level' + (level - 1)
          definitions['level' + level] = { [keyword]: [{ $ref }, { $ref }] }
        }
        const $ref = '#/definitions/level17'
        const schema =
          scope === 'root' ? { definitions, $ref } : { definitions, properties: { code: { $ref } } }
        expect(await read(client, schema)).toEqual([{ code: '123' }])
      },
    )
  }

  it.each(['root', 'property'])(
    `${client} still rejects recursive %s references after reading a shared branch`,
    async (scope) => {
      const $ref = '#/definitions/input'
      const leaf: JSONSchema7 =
        scope === 'root' ? { properties: { code: { type: 'string' } } } : { type: 'string' }
      const definitions = { input: { anyOf: [leaf, { $ref }] } }
      const schema =
        scope === 'root' ? { definitions, $ref } : { definitions, properties: { code: { $ref } } }
      await expect(read(client, schema)).rejects.toThrow('recursive tool schema')
    },
  )

  it.each([
    {
      anyOf: [
        { properties: { code: { type: 'string' } } },
        { properties: { code: { type: 'null' } } },
      ],
    },
    {
      oneOf: [
        { properties: { code: { const: '123' } }, required: ['code'] },
        { properties: { code: { const: '456' } }, required: ['code'] },
      ],
    },
    { anyOf: [{ properties: { code: { type: 'string' } } }, { additionalProperties: false }] },
    { anyOf: [false, { properties: { code: { type: 'string' } } }] },
    {
      anyOf: [{ $ref: '#/definitions/input' }, false],
      definitions: { input: { properties: { code: { type: 'string' } } } },
    },
    {
      allOf: [
        { anyOf: [{ properties: { code: { type: 'string' } } }, false] },
        { properties: { code: { type: ['string', 'integer'] } } },
      ],
    },
    {
      anyOf: [
        { properties: { code: { allOf: [{ type: 'number' }, { type: 'string' }] } } },
        { properties: { code: { type: 'string' } } },
      ],
    },
    { allOf: [true, { properties: { code: { type: 'string' } } }] },
    { properties: { code: { anyOf: [false, { type: 'string' }] } } },
    {
      anyOf: [{ $ref: '#/definitions/disabled' }, { properties: { code: { type: 'string' } } }],
      definitions: { disabled: false },
    },
    {
      definitions: { disabled: false },
      properties: { code: { anyOf: [{ $ref: '#/definitions/disabled' }, { type: 'string' }] } },
    },
  ] satisfies JSONSchema7[])(
    `${client} reads an unambiguous combined tool schema: %j`,
    async (schema) => {
      expect(await read(client, schema)).toEqual([{ code: '123' }])
    },
  )

  it.each([
    { properties: { code: false } },
    { anyOf: [false, { additionalProperties: false }] },
    { anyOf: [{ properties: { code: { type: 'string' } } }, {}] },
    {
      anyOf: [
        { properties: { code: { type: 'string' } } },
        { properties: { code: { type: 'number' } } },
      ],
    },
  ] satisfies JSONSchema7[])(
    `${client} rejects forbidden or ambiguous combined arguments: %j`,
    async (schema) => {
      await expect(read(client, schema)).rejects.toThrow('tool argument')
    },
  )

  it.each([
    { patternProperties: { '^code$': { type: 'string' } } },
    { patternProperties: { od: { type: 'string' } }, additionalProperties: false },
    {
      patternProperties: { '^code$': { type: 'string' } },
      additionalProperties: { type: 'number' },
    },
    {
      properties: { code: { type: ['string', 'number'] } },
      patternProperties: { '^code$': { type: 'string' } },
    },
    {
      patternProperties: { '^c': { type: ['string', 'number'] }, e$: { type: 'string' } },
    },
    {
      definitions: { text: { type: 'string' } },
      patternProperties: { '^code$': { $ref: '#/definitions/text' } },
    },
  ] satisfies JSONSchema7[])(
    `${client} reads strings matched by property patterns: %j`,
    async (schema) => {
      expect(await read(client, schema)).toEqual([{ code: '123' }])
    },
  )

  it.each([
    {
      patternProperties: { '^other$': { type: 'string' } },
      additionalProperties: { type: 'number' },
    },
    { patternProperties: { '^code$': true }, additionalProperties: false },
  ] satisfies JSONSchema7[])(`${client} keeps numeric patterned arguments: %j`, async (schema) => {
    expect(await read(client, schema)).toEqual([{ code: 123 }])
  })

  it.each([
    {
      properties: { code: { type: 'number' } },
      patternProperties: { '^code$': { type: 'string' } },
    },
    { patternProperties: { '^c': { type: 'number' }, e$: { type: 'string' } } },
    { patternProperties: { '^code$': false } },
    { patternProperties: { '^other$': true }, additionalProperties: false },
  ] satisfies JSONSchema7[])(
    `${client} rejects conflicting property patterns: %j`,
    async (schema) => {
      await expect(read(client, schema)).rejects.toThrow('tool argument')
    },
  )

  it(`${client} rejects an invalid property pattern`, async () => {
    await expect(read(client, { patternProperties: { '[': { type: 'string' } } })).rejects.toThrow(
      'tool schema pattern',
    )
  })

  for (const keyword of ['anyOf', 'oneOf'] as const) {
    const branches = [{ pattern: '^1' }, { pattern: '^4' }]
    it.each([
      { properties: { code: { type: 'string', [keyword]: branches } } },
      {
        properties: { code: { type: 'string' } },
        [keyword]: branches.map((code) => ({ properties: { code } })),
      },
      { properties: { code: { allOf: [{ type: 'string' }, { [keyword]: branches }] } } },
      {
        definitions: { text: { type: 'string' } },
        properties: { code: { $ref: '#/definitions/text', [keyword]: branches } },
      },
    ] satisfies JSONSchema7[])(
      `${client} keeps a declared string type through ${keyword} constraints: %j`,
      async (schema) => {
        expect(await read(client, schema)).toEqual([{ code: '123' }])
      },
    )

    it(`${client} keeps a declared numeric type through ${keyword} constraints`, async () => {
      const code: JSONSchema7 = { type: 'number', [keyword]: [{ minimum: 100 }, { maximum: -1 }] }
      expect(await read(client, { properties: { code } })).toEqual([{ code: 123 }])
    })

    it(`${client} keeps a declared object type through ${keyword} constraints`, async () => {
      const code: JSONSchema7 = {
        type: 'object',
        [keyword]: ['first', 'second'].map((kind) => ({ properties: { kind: { const: kind } } })),
      }
      expect(await read(client, { properties: { code } }, '{"kind":"first"}')).toEqual([
        { code: { kind: 'first' } },
      ])
    })

    it.each([
      { type: 'null', text: 'null', expected: null },
      { type: 'boolean', text: 'false', expected: false },
      { type: 'array', text: '["123"]', expected: ['123'] },
    ] as const)(
      `${client} keeps a declared $type through untyped ${keyword} branches`,
      async ({ type, text, expected }) => {
        const code: JSONSchema7 = { type, [keyword]: [true, false] }
        expect(await read(client, { properties: { code } }, text)).toEqual([{ code: expected }])
      },
    )

    it(`${client} still rejects an ambiguous ${keyword} with unconstrained branches`, async () => {
      const code: JSONSchema7 = { [keyword]: [{ type: 'string' }, {}] }
      await expect(read(client, { properties: { code } })).rejects.toThrow('tool argument')
    })
  }

  it.each([
    { oneOf: [{ const: '123' }, { const: '456' }] },
    { oneOf: [{ enum: ['123', '456'] }, { enum: ['789'] }] },
    {
      oneOf: [
        { type: 'string', pattern: '^1' },
        { type: 'string', pattern: '^4' },
      ],
    },
    {
      definitions: { first: { const: '123' }, second: { const: '456' } },
      oneOf: [
        { $ref: '#/properties/code/definitions/first' },
        { $ref: '#/properties/code/definitions/second' },
      ],
    },
  ] satisfies JSONSchema7[])(
    `${client} preserves string values across exclusive branches: %j`,
    async (code) => {
      expect(await read(client, { properties: { code } })).toEqual([{ code: '123' }])
    },
  )

  it(`${client} preserves numeric values across exclusive branches`, async () => {
    const code: JSONSchema7 = {
      oneOf: [
        { type: 'number', maximum: 0 },
        { type: 'number', minimum: 1 },
      ],
    }
    expect(await read(client, { properties: { code } })).toEqual([{ code: 123 }])
  })

  it(`${client} preserves object values across exclusive branches`, async () => {
    const code: JSONSchema7 = {
      oneOf: ['first', 'second'].map((kind) => ({
        type: 'object',
        properties: { kind: { const: kind } },
        required: ['kind'],
      })),
    }
    expect(await read(client, { properties: { code } }, '{"kind":"first"}')).toEqual([
      { code: { kind: 'first' } },
    ])
  })

  it(`${client} still rejects ambiguous strings or numbers across exclusive branches`, async () => {
    const code: JSONSchema7 = { oneOf: [{ type: 'string' }, { type: 'number' }] }
    await expect(read(client, { properties: { code } })).rejects.toThrow('ambiguous tool argument')
  })

  it.each(cases)(`${client} resolves a schema reference to $name`, async (test) => {
    expect(await read(client, test.schema, test.text)).toEqual([{ code: test.expected ?? '123' }])
  })

  it.each([
    '#/definitions/__proto__',
    '#/definitions/code~2text',
    '#/definitions/code/anyOf/00',
    '#/definitions/code/anyOf/length',
    '#/definitions/code/anyOf/-',
    '#/%ZZ',
  ])(`${client} rejects an invalid schema reference: %s`, async ($ref) => {
    await expect(
      read(client, {
        definitions: { 'code~2text': { type: 'string' }, code: { anyOf: [{ type: 'string' }] } },
        properties: { code: { $ref } },
      }),
    ).rejects.toThrow('cannot resolve a tool schema reference')
  })
}
