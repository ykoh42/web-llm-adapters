import { ModelError, type JSONValue } from './types.js'

/** Reject numbers that JSON.stringify would silently replace with null. */
export function parseJSON(source: string): JSONValue {
  return JSON.parse(source, (_key, value: JSONValue) => {
    if (typeof value === 'number' && !Number.isFinite(value))
      throw new ModelError('Tool arguments must contain finite numbers.')
    return value
  })
}
