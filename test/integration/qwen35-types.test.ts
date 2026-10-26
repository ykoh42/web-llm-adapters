import { Type, validateToolArguments } from '@earendil-works/pi-ai'
import { expect, it } from 'vitest'
import { readValue } from '../../shared/web-llm/qwen35/value.js'
import type { JSONSchema } from '../../shared/web-llm/types.js'

const cases = [
  { schema: { type: 'boolean' }, text: 'False', value: false },
  { schema: { type: 'boolean' }, text: 'TRUE', value: true },
  { schema: { type: 'boolean' }, text: '0', value: false },
  { schema: { type: 'boolean' }, text: '1', value: true },
  { schema: { type: ['integer', 'boolean'] }, text: '0', value: 0 },
  { schema: { type: ['integer', 'boolean'] }, text: 'False', value: false },
  { schema: { type: 'string' }, text: 'False', value: 'False' },
  { schema: { type: ['string', 'boolean'] }, text: 'False', value: 'False' },
  {
    schema: {
      type: ['string', 'null'],
      not: { anyOf: [{ type: 'number' }, { type: 'null' }] },
    },
    text: 'null',
    value: 'null',
  },
  {
    schema: { not: { allOf: [{ type: 'number' }, { type: 'integer' }] } },
    text: '123',
    value: '123',
  },
  {
    schema: { type: ['string', 'boolean'], not: { not: { type: 'string' } } },
    text: 'true',
    value: 'true',
  },
  {
    schema: {
      type: ['string', 'object'],
      not: { anyOf: [{ type: 'object' }, { type: 'array' }] },
    },
    text: '{"a":1}',
    value: '{"a":1}',
  },
  { schema: { not: { type: 'number' } }, text: '123', value: '123' },
  { schema: { not: { type: 'integer' } }, text: '123', value: '123' },
  { schema: { not: { type: 'boolean' } }, text: 'true', value: 'true' },
  { schema: { not: { type: 'null' } }, text: 'null', value: 'null' },
  { schema: { not: { type: 'object' } }, text: '{"a":1}', value: '{"a":1}' },
  { schema: { not: { type: 'array' } }, text: '[1,2]', value: '[1,2]' },
  { schema: { type: 'number', not: { type: 'integer' } }, text: '1.5', value: 1.5 },
  { schema: { type: 'number', not: { type: 'string' } }, text: '123', value: 123 },
  {
    schema: { not: { type: 'number', description: 'A numeric code.' } },
    text: '123',
    value: '123',
  },
  { schema: { not: { $ref: '#/definitions/number' } }, text: '123', value: '123' },
  { schema: { type: 'number', not: { type: 'number', minimum: 0 } }, text: '-123', value: -123 },
  { schema: { type: 'string', not: { enum: [123] } }, text: '123', value: '123' },
  { schema: { not: false }, text: '123', value: 123 },
  { schema: { type: ['string', 'null'], not: { enum: [null] } }, text: 'Seoul', value: 'Seoul' },
  { schema: { type: 'null', not: { enum: ['null'] } }, text: 'null', value: null },
  { schema: { allOf: [{ type: 'string' }, { not: { enum: [123] } }] }, text: '123', value: '123' },
  { schema: { allOf: [{ not: { enum: [123] } }, { type: 'string' }] }, text: '123', value: '123' },
  { schema: { type: 'string', allOf: [{ not: { enum: [123] } }] }, text: '123', value: '123' },
  { schema: { type: 'string', $ref: '#/definitions/exclusion' }, text: '123', value: '123' },
  {
    schema: { type: 'string', anyOf: [{ not: { enum: [123] } }, { type: 'number' }] },
    text: '123',
    value: '123',
  },
  {
    schema: { anyOf: [false, { type: 'string' }] },
    text: '{"code":123}',
    value: '{"code":123}',
  },
  {
    schema: { anyOf: [{ type: 'object' }, { type: 'null' }] },
    text: 'null',
    value: null,
  },
  {
    schema: { oneOf: [{ type: 'array', items: { type: 'string' } }, { type: 'null' }] },
    text: '["A7"]',
    value: ['A7'],
  },
  { schema: { enum: [7, 9] }, text: '7', value: 7 },
  { schema: { const: false }, text: 'false', value: false },
  { schema: { const: null }, text: 'null', value: null },
  { schema: { enum: [{ code: 'A7' }] }, text: '{"code":"A7"}', value: { code: 'A7' } },
] satisfies { schema: JSONSchema; text: string; value: unknown }[]

it.each(cases)('preserves a valid value with $schema: $text', ({ schema, text, value }) => {
  const parameters = Type.Object(
    { code: Type.Any(schema) },
    { definitions: { number: { type: 'number' }, exclusion: { not: { enum: [123] } } } },
  )
  const inputSchema: JSONSchema = JSON.parse(JSON.stringify(parameters))
  const tool = { name: 'lookup', description: '', parameters }
  const expected = {
    type: 'toolCall' as const,
    id: 'call',
    name: 'lookup',
    arguments: { code: value },
  }
  expect(validateToolArguments(tool, expected)).toEqual({ code: value })
  const code = readValue(text, { name: tool.name, description: '', inputSchema }, 'code')
  expect(code).toEqual(value)
  expect(validateToolArguments(tool, { ...expected, arguments: { code } })).toEqual({ code: value })
})

it.each(['yes', 'no', 'on', 'off', '1.0', '0.0', '"False"', 'null'])(
  'rejects an unsupported boolean spelling: %s',
  (text) => {
    expect(() =>
      readValue(
        text,
        {
          name: 'lookup',
          description: '',
          inputSchema: { properties: { code: { type: 'boolean' } } },
        },
        'code',
      ),
    ).toThrow()
  },
)

it.each([true, {}])('rejects a schema that excludes every value: %j', (not) => {
  expect(() =>
    readValue(
      '123',
      { name: 'lookup', description: '', inputSchema: { properties: { code: { not } } } },
      'code',
    ),
  ).toThrow('incompatible tool argument types')
})

it.each([
  { schema: { not: { enum: [123] } }, text: '123' },
  { schema: { type: ['string', 'null'], not: { enum: [null] } }, text: 'null' },
  { schema: { anyOf: [{ not: { enum: [null] } }, false] }, text: 'null' },
  {
    schema: {
      type: ['string', 'null'],
      not: { anyOf: [{ type: 'number' }, { const: null }] },
    },
    text: 'null',
  },
  {
    schema: {
      type: ['string', 'null'],
      not: { allOf: [{ type: 'number' }, { minimum: 0 }] },
    },
    text: 'null',
  },
  { schema: { not: { not: { enum: [null] } } }, text: 'null' },
] satisfies { schema: JSONSchema; text: string }[])(
  'rejects an unresolved exclusion instead of guessing: $schema',
  ({ schema, text }) => {
    expect(() =>
      readValue(
        text,
        {
          name: 'lookup',
          description: '',
          inputSchema: { properties: { code: schema } },
        },
        'code',
      ),
    ).toThrow('cannot determine a negated tool argument type')
  },
)

it.each([
  { schema: { not: false }, value: 123 },
  { schema: { not: { type: 'number' } }, value: 123 },
  { schema: { not: { type: 'number', minimum: 0 } }, value: 123 },
  { schema: { not: { type: ['number', 'string'], minimum: 0 } }, value: 123 },
  {
    schema: {
      properties: { code: { type: 'string' } },
      not: { properties: { code: { type: 'number' } }, required: ['code'] },
    },
    value: '123',
  },
] satisfies { schema: JSONSchema; value: unknown }[])(
  'preserves values with an object exclusion: $schema',
  ({ schema, value }) => {
    const parameters = Type.Object({}, schema)
    const inputSchema: JSONSchema = JSON.parse(JSON.stringify(parameters))
    const tool = { name: 'lookup', description: '', parameters }
    const call = {
      type: 'toolCall' as const,
      id: 'call',
      name: tool.name,
      arguments: { code: value },
    }
    expect(validateToolArguments(tool, call)).toEqual({ code: value })
    const code = readValue('123', { name: tool.name, description: '', inputSchema }, 'code')
    expect(code).toBe(value)
    expect(validateToolArguments(tool, { ...call, arguments: { code } })).toEqual({ code: value })
  },
)

it.each([
  { not: { properties: { code: { type: 'number' } }, required: ['code'] } },
  { allOf: [{ not: { properties: { code: { type: 'number' } }, required: ['code'] } }] },
] satisfies JSONSchema[])(
  'rejects an unresolved object exclusion instead of guessing: %j',
  (schema) => {
    const parameters = Type.Object({}, schema)
    const inputSchema: JSONSchema = JSON.parse(JSON.stringify(parameters))
    const tool = { name: 'lookup', description: '', parameters }
    const call = {
      type: 'toolCall' as const,
      id: 'call',
      name: tool.name,
      arguments: { code: '123' },
    }
    expect(validateToolArguments(tool, call)).toEqual({ code: '123' })
    expect(() =>
      readValue('123', { name: tool.name, description: '', inputSchema }, 'code'),
    ).toThrow('cannot determine a negated tool argument type')
  },
)

it.each([
  true,
  {},
  { type: 'object' },
  { anyOf: [{ type: 'array' }, { type: 'object' }] },
  { not: { type: 'number' } },
] satisfies (JSONSchema | boolean)[])('rejects an exclusion of every object: %j', (not) => {
  expect(() =>
    readValue('123', { name: 'lookup', description: '', inputSchema: { not } }, 'code'),
  ).toThrow('incompatible tool argument types')
})

it('rejects recursive references inside a negated type union', () => {
  const inputSchema: JSONSchema = {
    properties: { code: { not: { anyOf: [{ $ref: '#/definitions/loop' }] } } },
    definitions: { loop: { allOf: [{ $ref: '#/definitions/loop' }] } },
  }
  expect(() => readValue('null', { name: 'lookup', description: '', inputSchema }, 'code')).toThrow(
    'recursive tool schema',
  )
})

it.each(['allOf', 'not'] as const)('rejects cyclic in-memory schemas through %s', (keyword) => {
  const code: Record<string, unknown> = {}
  code[keyword] = keyword === 'allOf' ? [code] : code
  const inputSchema = { properties: { code } } as JSONSchema

  expect(() => readValue('null', { name: 'lookup', description: '', inputSchema }, 'code')).toThrow(
    'recursive tool schema',
  )
})

it('rejects cyclic schema roots', () => {
  const inputSchema: Record<string, unknown> = { properties: { code: { type: 'string' } } }
  inputSchema.allOf = [inputSchema]

  expect(() =>
    readValue(
      'text',
      {
        name: 'lookup',
        description: '',
        inputSchema: inputSchema as JSONSchema,
      },
      'code',
    ),
  ).toThrow('recursive tool schema')
})
