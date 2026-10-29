import { ModelError, type JSONValue } from './types.js'

/** Reject numbers that JSON.stringify would silently replace with null. */
export function parseJSON(source: string): JSONValue {
  const result: JSONValue = JSON.parse(source)
  const pending = [[result]]
  while (pending.length)
    for (const value of pending.pop()!) {
      if (typeof value === 'number' && !Number.isFinite(value))
        throw new ModelError('Tool arguments must contain finite numbers.')
      if (value && typeof value === 'object')
        pending.push(Array.isArray(value) ? value : Object.values(value))
    }
  return result
}

/** Preserve negative zero and reject numbers JSON would silently replace with null. */
export function stringifyJSON(input: unknown): string {
  const raw = Reflect.get(JSON, 'rawJSON')
  const isRaw = typeof raw === 'function' ? undefined : Reflect.get(JSON, 'isRawJSON')
  const marker = '\u0000'
  const markerJSON = '"\\u0000"'
  const markers: boolean[] = []
  let hasNegativeZero = false
  const text = JSON.stringify(input, (_key, value: unknown) => {
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) throw new TypeError('JSON values must contain finite numbers.')
      if (Object.is(value, -0)) {
        if (typeof raw === 'function') return raw('-0')
        hasNegativeZero = true
        markers.push(true)
        return marker
      }
    }
    // Remember literal marker values, including existing raw JSON strings.
    if (
      value === marker ||
      (typeof isRaw === 'function' &&
        typeof value === 'object' &&
        value !== null &&
        isRaw(value) &&
        Reflect.get(value, 'rawJSON') === markerJSON)
    )
      markers.push(false)
    return value
  })
  if (text === undefined) throw new TypeError('JSON values must be serializable.')
  if (!hasNegativeZero) return text

  let index = 0
  // String tokens followed by a colon are keys, not values produced by the replacer.
  return text.replace(/"(?:\\.|[^"\\])*"/g, (part, offset: number) => {
    if (part !== markerJSON || text[offset + part.length] === ':') return part
    return markers[index++] ? '-0' : part
  })
}

/** Compare JSON values without depending on object key order. */
export function sameJSON(left: unknown, right: unknown, distinguishNegativeZero = false): boolean {
  const same = distinguishNegativeZero
    ? Object.is
    : (left: unknown, right: unknown) => left === right
  if (same(left, right)) return true
  return compareJSON(left, right, same)
}

type Equal = (left: unknown, right: unknown) => boolean
type Pair = [unknown, unknown]

function compareJSON(left: unknown, right: unknown, same: Equal): boolean {
  // Only unequal values are added to the stack.
  const pending: Pair[] = [[left, right]]
  while (pending.length) {
    const [left, right] = pending.pop()!
    if (Array.isArray(left)) {
      if (!queueArrays(left, right, pending, same)) return false
      continue
    }
    if (!queueObjects(left, right, pending, same)) return false
  }
  return true
}

function queueArrays(left: unknown[], right: unknown, pending: Pair[], same: Equal): boolean {
  if (!Array.isArray(right) || left.length !== right.length) return false
  for (let index = 0; index < left.length; index++) {
    const leftValue = left[index] === undefined ? null : left[index]
    const rightValue = right[index] === undefined ? null : right[index]
    if (!same(leftValue, rightValue)) pending.push([leftValue, rightValue])
  }
  return true
}

function queueObjects(left: unknown, right: unknown, pending: Pair[], same: Equal): boolean {
  if (!isRecord(left) || !isRecord(right)) return false
  const keys = Object.keys(left).filter((key) => left[key] !== undefined)
  const rightKeys = Object.keys(right).filter((key) => right[key] !== undefined)
  if (keys.length !== rightKeys.length) return false
  for (const key of keys) {
    if (!Object.hasOwn(right, key)) return false
    if (!same(left[key], right[key])) pending.push([left[key], right[key]])
  }
  return true
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
