import { expect, it, vi } from 'vitest'
import { parseJSON, sameJSON, stringifyJSON } from '../../shared/web-llm/json.js'

it('compares JSON objects without depending on key order', () => {
  expect(
    sameJSON(
      { nested: { first: 1, second: [true, null] } },
      { nested: { second: [true, null], first: 1 } },
    ),
  ).toBe(true)
  expect(sameJSON({ values: [1, 2] }, { values: [2, 1] })).toBe(false)
  expect(sameJSON({ value: 1 }, { value: 2 })).toBe(false)
})

it('compares optional values as JSON serialization does', () => {
  expect(sameJSON({ schema: undefined }, {})).toBe(true)
  expect(sameJSON([undefined], [null])).toBe(true)
})

it('can distinguish negative zero inside nested values', () => {
  const negative = { values: [1, { number: -0 }] }
  const positive = { values: [1, { number: 0 }] }
  expect(sameJSON(negative, negative, true)).toBe(true)
  expect(sameJSON(negative, structuredClone(negative), true)).toBe(true)
  expect(sameJSON(negative, positive)).toBe(true)
  expect(sameJSON(negative, positive, true)).toBe(false)
})

it.each([
  'null',
  'true',
  '-0',
  '1.7976931348623157e308',
  '-1.7976931348623157e308',
  '5e-324',
  '"1e999"',
  '[null,false,-0,1e300,1e-300,"Infinity"]',
  '{"__proto__":{"value":42},"constructor":null,"values":[-0,1e300]}',
  '{"value":1e999,"value":42}',
])('preserves native JSON values: %s', (source) => {
  expect(parseJSON(source)).toStrictEqual(JSON.parse(source))
})

it.each(['1e999', '-1e999', '[1,1e999]', '{"values":[{"value":-1e999}]}'])(
  'rejects overflowing numbers: %s',
  (source) => {
    expect(() => parseJSON(source)).toThrow('Tool arguments must contain finite numbers.')
  },
)

it.each(['', 'undefined', 'NaN', 'Infinity', '[1,]', '{"value":}'])(
  'keeps native syntax errors: %s',
  (source) => {
    expect(() => parseJSON(source)).toThrow(SyntaxError)
  },
)

it('keeps every value in a large nested argument', () => {
  const input = {
    items: Array.from({ length: 1000 }, (_, index) => ({
      index,
      values: [null, true, -0, 1e300, 1e-300, '서울 😀'],
    })),
  }
  // JSON encoding defines the negative-zero behavior before the parser sees it.
  const source = JSON.stringify(input)
  expect(parseJSON(source)).toStrictEqual(JSON.parse(source))
  expect(() => parseJSON(source.replace('1e+300', '1e999'))).toThrow('finite numbers')
})

it.each([0, 9999])('rejects an overflowing number at array index %s', (index) => {
  const values = Array<string>(10000).fill('42')
  values[index] = '1e999'
  expect(() => parseJSON('[' + values.join(',') + ']')).toThrow('finite numbers')
})

it('validates deeply nested arrays without using the call stack', () => {
  const wrap = (value: string) => '['.repeat(3000) + value + ']'.repeat(3000)
  const source = wrap('42')
  expect(JSON.stringify(parseJSON(source)) === source).toBe(true)
  expect(() => parseJSON(wrap('1e999'))).toThrow('finite numbers')
})

it.each(['native', 'without rawJSON'] as const)(
  'preserves negative zero while respecting JSON serialization (%s)',
  (mode) => {
    if (mode === 'without rawJSON')
      vi.stubGlobal('JSON', Object.assign(Object.create(JSON), { rawJSON: undefined }))
    try {
      let reads = 0
      const shared = { value: -0 }
      const input = {
        first: shared,
        second: shared,
        omitted: undefined,
        array: [undefined, -0, 0, { nested: -0 }],
        escaped: 'a "quote" and \n newline',
        converted: { toJSON: () => ({ value: -0 }) },
        get read() {
          reads++
          return -0
        },
        ['__proto__']: { nested: -0 },
      }
      const result = JSON.parse(stringifyJSON(input))
      expect(reads).toBe(1)
      expect(Object.is(result.first.value, -0)).toBe(true)
      expect(Object.is(result.second.value, -0)).toBe(true)
      expect(result.array.map((value: unknown) => Object.is(value, -0))).toEqual([
        false,
        true,
        false,
        false,
      ])
      expect(result.array[0]).toBeNull()
      expect(Object.is(result.array[3].nested, -0)).toBe(true)
      expect(Object.is(result.converted.value, -0)).toBe(true)
      expect(Object.is(result.read, -0)).toBe(true)
      expect(Object.is(result.__proto__.nested, -0)).toBe(true)
      expect(Object.getPrototypeOf(result)).toBe(Object.prototype)
      expect(result).not.toHaveProperty('omitted')
      expect(result.escaped).toBe(input.escaped)
      expect(Object.is(JSON.parse(stringifyJSON(-0)), -0)).toBe(true)
      expect(() => stringifyJSON({ value: Infinity })).toThrow('finite numbers')
    } finally {
      vi.unstubAllGlobals()
    }
  },
)

it('matches native negative-zero serialization without rawJSON', () => {
  const raw = Reflect.get(JSON, 'rawJSON') as (value: string) => unknown
  const values = Array.from({ length: 256 }, (_, index) => ({
    id: index,
    values: [index % 2 ? -0 : 0, index - 128, null, index % 3 === 0],
    nested: Object.fromEntries([
      ['__proto__', { value: index % 5 ? -0 : 0 }],
      ['text', `서울 ${index} "\\\n`],
      ['array', [undefined, -0, [index / 10]]],
    ]),
  }))
  const expected = values.map((value) =>
    JSON.stringify(value, (_key, item: unknown) => (Object.is(item, -0) ? raw('-0') : item)),
  )
  vi.stubGlobal('JSON', Object.assign(Object.create(JSON), { rawJSON: undefined }))
  try {
    expect(values.map(stringifyJSON)).toEqual(expected)
  } finally {
    vi.unstubAllGlobals()
  }
})

it('keeps literal marker strings and keys when JSON.rawJSON is unavailable', () => {
  const raw = Reflect.get(JSON, 'rawJSON') as (value: string) => unknown
  const marker = '\u0000'
  const input = {
    [marker]: marker,
    values: [marker, -0, raw(JSON.stringify(marker)), { [marker]: -0 }, marker, -0],
    text: 'a "\\u0000" string',
  }
  const expected = JSON.stringify(input, (_key, value: unknown) =>
    Object.is(value, -0) ? raw('-0') : value,
  )
  vi.stubGlobal('JSON', Object.assign(Object.create(JSON), { rawJSON: undefined }))
  try {
    expect(stringifyJSON(input)).toBe(expected)
  } finally {
    vi.unstubAllGlobals()
  }
})
