import { ModelError, type ToolSpec } from '../types.js'
import { parseJSON } from '../json.js'
import { argumentSchema, type ArgumentSchemaCache } from './schema.js'

export function readValue(
  source: string,
  tool: ToolSpec,
  name: string,
  cache?: ArgumentSchemaCache,
) {
  const { types, unknownNegation } = argumentSchema(tool, name, cache)
  const lineEnding = source.startsWith('\r\n') ? '\r\n' : source.startsWith('\n') ? '\n' : ''
  let text = source
  if (lineEnding) {
    // Match the closing delimiter to the opener so a value's final `\r` survives.
    text = text.slice(lineEnding.length)
    const closing = lineEnding === '\r\n' && text.endsWith('\r\n') ? '\r\n' : '\n'
    if (text.endsWith(closing)) text = text.slice(0, -closing.length)
  } else text = text.replace(/\r?\n$/, '')
  if (types?.includes('string')) return readString(text, types, unknownNegation)
  return readJSON(text, types)
}

function readString(text: string, types: string[], unknownNegation: boolean) {
  if (types.includes('null') && text.trim() === 'null') {
    if (unknownNegation)
      throw new ModelError('Qwen3.5 cannot determine a negated tool argument type.')
    return null
  }
  const otherTypes = types.filter((type) => type !== 'string' && type !== 'null')
  if (otherTypes.length) {
    try {
      // Inspect the candidate type without rejecting valid string contents.
      const value: unknown = JSON.parse(text)
      const overflow = typeof value === 'number' && !Number.isFinite(value)
      if (otherTypes.some((type) => matchesType(value, type) || (type === 'integer' && overflow)))
        throw new ModelError('Qwen3.5 returned an ambiguous tool argument.')
    } catch (cause) {
      if (cause instanceof ModelError) throw cause
    }
  }
  return text
}

function readJSON(text: string, types: string[] | undefined) {
  // Qwen's schema-aware parser accepts case-insensitive booleans and 0/1.
  const booleanText = types?.includes('boolean') ? text.trim().toLowerCase() : undefined
  if (booleanText === 'true' || booleanText === 'false') return booleanText === 'true'
  try {
    const value = parseJSON(text)
    if (types?.length && !types.some((type) => matchesType(value, type))) {
      if (booleanText === '0' || booleanText === '1') return booleanText === '1'
      throw new ModelError('Qwen3.5 returned a tool argument with the wrong type.')
    }
    return value
  } catch (cause) {
    if (cause instanceof ModelError) throw cause
    if (types === undefined) return text
    throw new ModelError('Qwen3.5 returned an invalid tool argument.', { cause })
  }
}

function matchesType(value: unknown, type: string) {
  if (type === 'integer') return typeof value === 'number' && Number.isInteger(value)
  if (type === 'nonInteger') return typeof value === 'number' && !Number.isInteger(value)
  if (type === 'null') return value === null
  if (type === 'array') return Array.isArray(value)
  if (type === 'object') return isRecord(value)
  return typeof value === type
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
