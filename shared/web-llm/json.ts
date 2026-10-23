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
