import { expect, it } from 'vitest'
import { parseJSON, sameJSON } from '../../shared/web-llm/json.js'

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
