import { ModelError } from '../types.js'
import { parseJSON } from '../json.js'

const quote = '<|"|>'

function writeKey(key: string): string {
  return /^[\p{L}\p{N}_$.-]+$/u.test(key) ? key : writeValue(key)
}

type Piece = string | { value: object; close?: string }

function fields(value: object): [string, unknown][] {
  return Object.entries(value)
    .filter(([, item]) => item !== undefined)
    .sort(([left], [right]) => compareKeys(left, right))
}

// Jinja's dictsort uses lowercase keys and Python's Unicode code-point order.
function compareKeys(left: string, right: string) {
  left = left.toLowerCase()
  right = right.toLowerCase()
  let leftIndex = 0
  let rightIndex = 0
  while (leftIndex < left.length && rightIndex < right.length) {
    const leftPoint = left.codePointAt(leftIndex)!
    const rightPoint = right.codePointAt(rightIndex)!
    if (leftPoint !== rightPoint) return leftPoint - rightPoint
    leftIndex += leftPoint > 0xffff ? 2 : 1
    rightIndex += rightPoint > 0xffff ? 2 : 1
  }
  return left.length - right.length
}

// This tracks only synchronous calls and is restored by finally.
let writeDepth = 0

/** Gemma uses a special string delimiter and sorted bare object keys. */
export function writeValue(value: unknown): string {
  if (typeof value === 'string') {
    // Keep literal control markers inside returned strings.
    if (value.includes('<|') || value.includes('|>'))
      return JSON.stringify(value).replaceAll('<', '\\u003c')
    return quote + value + quote
  }
  if (typeof value === 'number' && Number.isFinite(value))
    return Object.is(value, -0) ? '-0' : String(value)
  if (value === null || typeof value === 'boolean') return String(value)
  if (typeof value !== 'object') throw new ModelError('Gemma tool values must be JSON values.')
  // Switch to a work list before exhausting the JavaScript stack.
  if (writeDepth >= 32) return writeDeep(value, writeValue)
  writeDepth++
  try {
    if (Array.isArray(value)) return writeArray(value)
    return (
      '{' +
      fields(value)
        .map(([key, item]) => writeKey(key) + ':' + writeValue(item))
        .join(',') +
      '}'
    )
  } finally {
    writeDepth--
  }
}

function writeArray(value: unknown[]): string {
  const items: string[] = []
  for (let index = 0; index < value.length; index++) {
    if (!(index in value)) throw new ModelError('Gemma tool values must be JSON values.')
    items.push(writeValue(value[index]))
  }
  return '[' + items.join(',') + ']'
}

function writeDeep(value: object, writeScalar: (value: unknown) => string): string {
  function valuePiece(value: unknown): Piece {
    return typeof value === 'object' && value !== null ? { value } : writeScalar(value)
  }
  const pending: Piece[] = [{ value }]
  const active = new WeakSet<object>()
  let text = ''
  while (pending.length) {
    const piece = pending.pop()!
    if (typeof piece === 'string') {
      text += piece
      continue
    }
    const { value, close } = piece
    if (close) {
      text += close
      active.delete(value)
      continue
    }
    if (active.has(value)) throw new ModelError('Gemma tool values must be JSON values.')
    active.add(value)
    if (Array.isArray(value)) {
      text += '['
      queueArray(value, pending, valuePiece)
    } else {
      text += '{'
      queueObject(value, pending, valuePiece)
    }
  }
  return text
}

function queueArray(value: unknown[], pending: Piece[], valuePiece: (value: unknown) => Piece) {
  pending.push({ value, close: ']' })
  for (let index = value.length - 1; index >= 0; index--) {
    if (!(index in value)) throw new ModelError('Gemma tool values must be JSON values.')
    pending.push(valuePiece(value[index]))
    if (index) pending.push(',')
  }
}

function queueObject(value: object, pending: Piece[], valuePiece: (value: unknown) => Piece) {
  const entries = fields(value)
  pending.push({ value, close: '}' })
  for (let index = entries.length - 1; index >= 0; index--) {
    const [key, item] = entries[index]!
    pending.push(valuePiece(item), writeKey(key) + ':')
    if (index) pending.push(',')
  }
}

/** Read one complete argument object, without evaluating generated code. */
export function readArguments(source: string) {
  // Keep quoted strings whole and capture bare keys with their colon.
  const token = /\s*(<\|"\|>[\s\S]*?<\|"\|>|"(?:\\.|[^"\\])*"|[{}[\]:,]|([^\s{}[\]:,]+)(\s*:)?)/gy
  const parts: string[] = []
  let depth = 0
  for (;;) {
    const match = token.exec(source)
    if (!match) throw new ModelError('Gemma returned incomplete tool arguments.')
    const value = match[2] ?? match[1]!
    if (!parts.length && value !== '{')
      throw new ModelError('Gemma tool arguments must be an object.')
    if (value === '{' || value === '[') depth++
    if (value === '}' || value === ']') depth--
    addArgument(parts, value, Boolean(match[3]))
    if (depth !== 0) continue
    return { input: argumentObject(parts.join('')), end: token.lastIndex }
  }
}

function addArgument(parts: string[], value: string, key: boolean): void {
  if (value.startsWith(quote)) {
    if (value.length < quote.length * 2 || !value.endsWith(quote))
      throw new ModelError('Gemma returned an unclosed tool string.')
    parts.push(JSON.stringify(value.slice(quote.length, -quote.length)))
  } else if (key) {
    // A missing JSON quote must remain an error.
    parts.push(value.startsWith('"') ? value : JSON.stringify(value), ':')
  } else parts.push(value)
}

function argumentObject(source: string) {
  try {
    const input = parseJSON(source)
    if (input === null || Array.isArray(input) || typeof input !== 'object')
      throw new Error('Expected an object')
    return input
  } catch (cause) {
    throw new ModelError('Gemma returned invalid tool arguments.', { cause })
  }
}
