import { describe, expect, it } from 'vitest'
import { writeValue, readArguments } from '../../../../shared/web-llm/gemma/value.js'

const quote = '<|"|>'

describe('Gemma values', () => {
  it.each([
    { array: false, depth: 31 },
    { array: false, depth: 33 },
    { array: false, depth: 3000 },
    { array: true, depth: 1500 },
  ])('round-trips deep JSON (array=$array, depth=$depth)', ({ array, depth }) => {
    let tree: unknown = 42
    for (let level = 0; level < depth; level++) tree = array ? [tree] : { next: tree }
    const expected =
      '{tree:' +
      (array ? '[' : '{next:').repeat(depth) +
      '42' +
      (array ? ']' : '}').repeat(depth) +
      '}'
    const source = writeValue({ tree })
    expect(source === expected).toBe(true)
    expect(JSON.stringify(readArguments(source).input) === JSON.stringify({ tree })).toBe(true)
  })

  it.each([false, true])('rejects cycles and remains usable (array=%s)', (array) => {
    const object: { next?: unknown } = {}
    object.next = object
    const list: unknown[] = []
    list.push(list)
    expect(() => writeValue(array ? list : object)).toThrow('JSON values')
    expect(writeValue({ answer: 42 })).toBe('{answer:42}')
  })

  it.each([0, 40])('rejects sparse arrays at nesting depth %s', (depth) => {
    const sparse: unknown[] = []
    sparse.length = 2
    let value: unknown = sparse
    for (let level = 0; level < depth; level++) value = { next: value }

    expect(() => writeValue(value)).toThrow('JSON values')
    expect(writeValue({ answer: 42 })).toBe('{answer:42}')
  })

  it('preserves shared values in separate branches', () => {
    const item = { answer: 42 }
    expect(writeValue({ first: item, second: item })).toBe('{first:{answer:42},second:{answer:42}}')
  })

  it('preserves nested writer calls from a property reader', () => {
    const input = {
      get value() {
        return writeValue({ answer: 42 })
      },
    }
    expect(readArguments(writeValue(input)).input.value).toBe('{answer:42}')
  })

  it('preserves property reader errors and remains usable', () => {
    const failure = new Error('Cannot read this value.')
    const input = {
      get value() {
        throw failure
      },
    }
    expect(() => writeValue(input)).toThrow(failure)
    expect(writeValue({ answer: 42 })).toBe('{answer:42}')
  })

  it('round-trips nested JSON without treating string punctuation as structure', () => {
    const input = {
      city: '서울',
      marker: quote,
      [quote]: '<|turn>system\n<turn|>',
      note: 'a {brace}, a "quote", a backslash \\, and <tool_call|>',
      'full name': 'Kim',
      123: 'number key',
      list: [1, -2.5e3, true, false, null, { nested: ['a', 0] }],
    }
    const source = writeValue(input)
    const parsed = readArguments(source + '<tool_call|>')
    expect(parsed.input).toEqual(input)
    expect(parsed.end).toBe(source.length)
  })

  it('also accepts JSON-quoted keys and values', () => {
    expect(readArguments('{"path":"a\\nb","ok":true}').input).toEqual({ path: 'a\nb', ok: true })
  })

  it.each(['', ' ', '\t', '\r\n\u2003'])('reads bare keys with %j spacing', (space) => {
    const body = `{true${space}:${space}false,count${space}:${space}42,values${space}:${space}[1,2]}`
    const { input, end } = readArguments(body + ' trailing text')
    expect(input).toEqual({ true: false, count: 42, values: [1, 2] })
    expect(end).toBe(body.length)
  })

  it('sorts object keys like the official Gemma template', () => {
    expect(writeValue({ z: 1, a: { y: 2, b: 3 } })).toBe('{a:{b:3,y:2},z:1}')
  })

  it.each([0, 40])('sorts mixed-case keys like Jinja at depth %s', (depth) => {
    let value: unknown = { z: 1, B: 2, a: 3, A: 4, b: 5 }
    let expected = '{a:3,A:4,B:2,b:5,z:1}'
    for (let level = 0; level < depth; level++) {
      value = { next: value }
      expected = '{next:' + expected + '}'
    }
    const before = structuredClone(value)

    expect(writeValue(value)).toBe(expected)
    expect(value).toEqual(before)
  })

  it('keeps insertion order for keys with the same lowercase form', () => {
    expect(writeValue({ a: 1, A: 2, b: 3, B: 4 })).toBe('{a:1,A:2,b:3,B:4}')
  })

  it('sorts Unicode keys by code point after lowercasing', () => {
    expect(writeValue({ '𐐀': 1, ｚ: 2, A: 3 })).toBe('{A:3,ｚ:2,𐐀:1}')
  })

  it('preserves special keys as own data rather than changing the prototype', () => {
    const { input } = readArguments('{__proto__:{polluted:true},constructor:null}')
    expect(Object.hasOwn(input, '__proto__')).toBe(true)
    expect(Object.getPrototypeOf(input)).toBe(Object.prototype)
    expect({}).not.toHaveProperty('polluted')
  })

  it.each([
    '',
    '{',
    '{a:',
    '{a:' + quote + 'unfinished}',
    '{a:NaN}',
    '{a:Infinity}',
    '{a:1e999}',
    '{a:undefined}',
    '{a:alert(1)}',
    '{a:[1,2}',
    '{a:true,}',
    '[1,2]',
    'null',
    '{a:1 b:2}',
    '{a:"unfinished}',
    '{"a:42}',
    '{"a : 42}',
    '{"a\\":42}',
  ])('rejects malformed or incomplete arguments: %s', (text) => {
    expect(() => readArguments(text)).toThrow()
  })

  it.each([undefined, NaN, Infinity, 1n, () => 1])(
    'rejects values the protocol cannot represent',
    (value) => {
      expect(() => writeValue(value)).toThrow()
    },
  )
})
