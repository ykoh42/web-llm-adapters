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

/** Keep invalid JSON numbers from silently becoming null. */
export function stringifyJSON(input: unknown): string {
  const text = JSON.stringify(input, (_key, value: unknown) => {
    if (typeof value === 'number' && !Number.isFinite(value))
      throw new TypeError('JSON values must contain finite numbers.')
    return value
  })
  if (text === undefined) throw new TypeError('JSON values must be serializable.')
  return text
}

/** Compare JSON values without depending on object key order. */
export function sameJSON(left: unknown, right: unknown): boolean {
  const pending: [unknown, unknown][] = [[left, right]]
  while (pending.length) {
    const [left, right] = pending.pop()!
    if (left === right) continue
    if (Array.isArray(left)) {
      if (!Array.isArray(right) || left.length !== right.length) return false
      for (let index = 0; index < left.length; index++) {
        const leftValue = left[index] === undefined ? null : left[index]
        const rightValue = right[index] === undefined ? null : right[index]
        if (leftValue !== rightValue) pending.push([leftValue, rightValue])
      }
      continue
    }
    if (!isRecord(left) || !isRecord(right)) return false
    const keys = Object.keys(left).filter((key) => left[key] !== undefined)
    const rightKeys = Object.keys(right).filter((key) => right[key] !== undefined)
    if (keys.length !== rightKeys.length) return false
    for (const key of keys) {
      if (!Object.hasOwn(right, key)) return false
      if (left[key] !== right[key]) pending.push([left[key], right[key]])
    }
  }
  return true
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
