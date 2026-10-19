import { ModelError, type JSONValue } from '../types.js'

const quote = '<|"|>'

function writeKey(key: string): string {
  return /^[\p{L}\p{N}_$.-]+$/u.test(key) ? key : writeValue(key)
}

/** Gemma uses a special string delimiter and bare object keys. */
export function writeValue(value: unknown): string {
  if (typeof value === 'string') {
    // A tool may return source code containing literal control markers.
    if (value.includes('<|') || value.includes('|>'))
      return JSON.stringify(value).replaceAll('<', '\\u003c')
    return quote + value + quote
  }
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (value === null || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) return '[' + value.map(writeValue).join(',') + ']'
  if (typeof value === 'object' && value !== null) {
    const fields = Object.entries(value)
      .filter(([, item]) => item !== undefined)
      .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
      .map(([key, item]) => writeKey(key) + ':' + writeValue(item))
    return '{' + fields.join(',') + '}'
  }
  throw new ModelError('Gemma tool values must be JSON values.')
}

/** Read one complete argument object, without evaluating generated code. */
export function readArguments(source: string) {
  // Match whole strings so their braces and control markers stay literal data.
  const token = /\s*(<\|"\|>[\s\S]*?<\|"\|>|"(?:\\.|[^"\\])*"|[{}[\]:,]|[^\s{}[\]:,]+)/gy
  const parts: string[] = []
  let depth = 0
  for (;;) {
    const match = token.exec(source)
    if (!match) throw new ModelError('Gemma returned incomplete tool arguments.')
    const value = match[1]!
    if (!parts.length && value !== '{')
      throw new ModelError('Gemma tool arguments must be an object.')
    if (value === '{' || value === '[') depth++
    if (value === '}' || value === ']') depth--
    if (value.startsWith(quote)) {
      if (value.length < quote.length * 2 || !value.endsWith(quote))
        throw new ModelError('Gemma returned an unclosed tool string.')
      parts.push(JSON.stringify(value.slice(quote.length, -quote.length)))
    } else if (
      !value.startsWith('"') &&
      !/[{}[\]:,]/.test(value) &&
      /^\s*:/.test(source.slice(token.lastIndex))
    ) {
      parts.push(JSON.stringify(value))
    } else parts.push(value)
    if (depth !== 0) continue
    return { input: argumentObject(parts.join('')), end: token.lastIndex }
  }
}

function argumentObject(source: string) {
  try {
    const input: JSONValue = JSON.parse(source, (_key, item: JSONValue) => {
      if (typeof item === 'number' && !Number.isFinite(item)) throw new Error('Non-finite number')
      return item
    })
    if (input === null || Array.isArray(input) || typeof input !== 'object')
      throw new Error('Expected an object')
    return input
  } catch (cause) {
    throw new ModelError('Gemma returned invalid tool arguments.', { cause })
  }
}
