import { expect, it } from 'vitest'
import type { JSONSchema } from '../../shared/web-llm/types.js'
import { readValue } from '../../shared/web-llm/qwen35/value.js'

const cases: { schema: JSONSchema; text: string; expected: unknown }[] = [
  {
    schema: { properties: { code: { type: 'string', const: undefined } } },
    text: '123',
    expected: '123',
  },
  {
    schema: { properties: { code: { const: undefined } } },
    text: '123',
    expected: 123,
  },
  {
    schema: { properties: { code: { type: ['string', 'null'], not: undefined } } },
    text: 'null',
    expected: null,
  },
  {
    schema: { not: undefined, properties: { code: { type: ['string', 'null'] } } },
    text: 'null',
    expected: null,
  },
  {
    schema: {
      properties: {
        code: { type: ['string', 'null'], not: { type: 'number', minimum: undefined } },
      },
    },
    text: 'null',
    expected: null,
  },
]

it.each(cases)(
  'reads the same argument as the JSON payload: $schema',
  ({ schema, text, expected }) => {
    const tool = { name: 'lookup', description: '', inputSchema: schema }
    const payload = { ...tool, inputSchema: JSON.parse(JSON.stringify(schema)) }
    expect(readValue(text, payload, 'code')).toEqual(expected)
    expect(readValue(text, tool, 'code')).toEqual(expected)
  },
)

it.each([
  { properties: { code: { not: { type: undefined } } } },
  { properties: { code: { not: { not: undefined } } } },
] satisfies JSONSchema[])('keeps an empty negation forbidden after JSON encoding: %j', (schema) => {
  const tool = { name: 'lookup', description: '', inputSchema: schema }
  const payload = { ...tool, inputSchema: JSON.parse(JSON.stringify(schema)) }
  expect(() => readValue('123', payload, 'code')).toThrow('incompatible tool argument types')
  expect(() => readValue('123', tool, 'code')).toThrow('incompatible tool argument types')
})
