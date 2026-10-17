import { describe, expect, it } from 'vitest'
import { writeValue, readArguments } from '../../src/gemma/value.js'

const quote = '<|"|>'

describe('Gemma values', () => {
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
