import { ModelError, type ToolSpec } from '../types.js'
import { parseJSON } from '../json.js'

const jsonTypes = ['string', 'number', 'boolean', 'null', 'array', 'object']

interface SchemaContext {
  root: Record<string, unknown>
  name: string
  fields: Map<Record<string, unknown>, unknown[]>
  types: Map<Record<string, unknown>, string[] | undefined>
}

export function readValue(source: string, tool: ToolSpec, name: string) {
  const types = propertyTypes(tool, name)
  const text = source.replace(/^\r?\n/, '').replace(/\r?\n$/, '')
  if (types?.includes('string')) {
    if (types.includes('null') && text.trim() === 'null') return null
    const otherTypes = types.filter((type) => type !== 'string' && type !== 'null')
    if (otherTypes.length) {
      try {
        const value = parseJSON(text)
        if (otherTypes.some((type) => matchesType(value, type)))
          throw new ModelError('Qwen3.5 returned an ambiguous tool argument.')
      } catch (cause) {
        if (cause instanceof ModelError) throw cause
      }
    }
    return text
  }
  try {
    const value = parseJSON(text)
    if (types?.length && !types.some((type) => matchesType(value, type)))
      throw new ModelError('Qwen3.5 returned a tool argument with the wrong type.')
    return value
  } catch (cause) {
    if (cause instanceof ModelError) throw cause
    if (types === undefined) return text
    throw new ModelError('Qwen3.5 returned an invalid tool argument.', { cause })
  }
}

function propertyTypes(tool: ToolSpec, name: string): string[] | undefined {
  const schema = tool.inputSchema
  if (!isRecord(schema)) return undefined
  // Keep results within one argument read; schemas may change between requests.
  const context: SchemaContext = { root: schema, name, fields: new Map(), types: new Map() }
  const schemas = propertySchemas(schema, context)
  const types = schemas
    .map((value) => readTypes(value, context))
    .filter((value): value is string[] => value !== undefined)
  if (!types.length) return undefined

  const [first, ...rest] = types
  const intersection = rest.reduce(intersectTypes, first!)
  if (!intersection.length) throw new ModelError('Qwen3.5 found incompatible tool argument types.')
  return intersection
}

function intersectTypes(types: string[], next: string[]) {
  const result = new Set<string>()
  for (const type of types) {
    for (const candidate of next) {
      if (type === candidate) result.add(type)
      else if (type === 'integer' && candidate === 'number') result.add('integer')
      else if (type === 'number' && candidate === 'integer') result.add('integer')
    }
  }
  return [...result]
}

function propertySchemas(
  schema: unknown,
  context: SchemaContext,
  references = new Set<string>(),
): unknown[] {
  if (typeof schema === 'boolean') return [schema]
  if (!isRecord(schema)) throw new ModelError('Qwen3.5 cannot read a tool schema.')
  const cached = context.fields.get(schema)
  if (cached) return cached
  const schemas: unknown[] = []

  if (typeof schema.$ref === 'string') {
    if (references.has(schema.$ref)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
    const target = resolveReference(context.root, schema.$ref)
    if (typeof target !== 'boolean' && !isRecord(target))
      throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
    schemas.push(...propertySchemas(target, context, new Set([...references, schema.$ref])))
  }

  let matched = false
  if (isRecord(schema.properties) && Object.hasOwn(schema.properties, context.name)) {
    schemas.push(schema.properties[context.name])
    matched = true
  }
  if (isRecord(schema.patternProperties)) {
    try {
      for (const [pattern, value] of Object.entries(schema.patternProperties)) {
        if (!new RegExp(pattern, 'u').test(context.name)) continue
        schemas.push(value)
        matched = true
      }
    } catch (cause) {
      throw new ModelError('Qwen3.5 cannot read a tool schema pattern.', { cause })
    }
  }
  if (!matched) {
    if (schema.additionalProperties === false) schemas.push(false)
    else if (isRecord(schema.additionalProperties)) schemas.push(schema.additionalProperties)
  }

  for (const keyword of ['allOf', 'anyOf', 'oneOf'] as const) {
    const alternatives = schema[keyword]
    if (alternatives === undefined) continue
    if (!Array.isArray(alternatives) || !alternatives.length)
      throw new ModelError('Qwen3.5 cannot read a tool schema.')
    // Exclusive branches may share argument types; exclusivity applies to the whole object.
    const operation = keyword === 'allOf' ? 'allOf' : 'anyOf'
    schemas.push({
      [operation]: alternatives.map((branch) => ({
        allOf: propertySchemas(branch, context, references),
      })),
    })
  }

  context.fields.set(schema, schemas)
  return schemas
}

function readTypes(
  value: unknown,
  context: SchemaContext,
  references = new Set<string>(),
): string[] | undefined {
  if (value === false) return []
  if (!isRecord(value)) return undefined
  if (context.types.has(value)) return context.types.get(value)

  const constraints: string[][] = []
  if (typeof value.$ref === 'string') {
    if (references.has(value.$ref)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
    const target = resolveReference(context.root, value.$ref)
    if (typeof target !== 'boolean' && !isRecord(target))
      throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
    const types = readTypes(target, context, new Set([...references, value.$ref]))
    if (types) constraints.push(types)
  }

  const types = Array.isArray(value.type)
    ? value.type.filter((type): type is string => typeof type === 'string')
    : typeof value.type === 'string'
      ? [value.type]
      : []
  if (types.length) constraints.push(types)
  if (Array.isArray(value.enum) && value.enum.length)
    constraints.push([...new Set(value.enum.map(valueType))])
  if (Object.hasOwn(value, 'const')) constraints.push([valueType(value.const)])

  for (const keyword of ['anyOf', 'oneOf'] as const) {
    const alternatives = value[keyword]
    if (alternatives === undefined) continue
    if (!Array.isArray(alternatives) || !alternatives.length)
      throw new ModelError('Qwen3.5 cannot determine a tool argument type.')
    // An untyped branch admits every JSON type; sibling rules can still narrow it.
    const branchTypes = alternatives.map(
      (branch) => readTypes(branch, context, references) ?? jsonTypes,
    )
    constraints.push([...new Set(branchTypes.flat())])
  }

  if (value.allOf !== undefined) {
    if (!Array.isArray(value.allOf))
      throw new ModelError('Qwen3.5 cannot determine a tool argument type.')
    const branchTypes = value.allOf
      .map((branch) => readTypes(branch, context, references))
      .filter((branch): branch is string[] => branch !== undefined)
    constraints.push(...branchTypes)
  }

  if (!constraints.length) {
    context.types.set(value, undefined)
    return undefined
  }
  const [first, ...rest] = constraints
  const result = rest.reduce(intersectTypes, first!)
  context.types.set(value, result)
  return result
}

function valueType(value: unknown): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  if (typeof value === 'number' && Number.isInteger(value)) return 'integer'
  return typeof value
}

function resolveReference(root: Record<string, unknown>, reference: string): unknown {
  if (!reference.startsWith('#')) return undefined
  let pointer: string
  try {
    pointer = decodeURIComponent(reference.slice(1))
  } catch {
    return undefined
  }
  if (!pointer) return root
  if (!pointer.startsWith('/')) return undefined

  let value: unknown = root
  for (const token of pointer.slice(1).split('/')) {
    if (/~(?:[^01]|$)/.test(token)) return undefined
    const key = token.replaceAll('~1', '/').replaceAll('~0', '~')
    if (Array.isArray(value)) {
      if (!/^(0|[1-9]\d*)$/.test(key) || !Object.hasOwn(value, key)) return undefined
      value = value[Number(key)]
    } else {
      if (!isRecord(value) || !Object.hasOwn(value, key)) return undefined
      value = value[key]
    }
  }
  return value
}

function matchesType(value: unknown, type: string) {
  if (type === 'integer') return typeof value === 'number' && Number.isInteger(value)
  if (type === 'null') return value === null
  if (type === 'array') return Array.isArray(value)
  if (type === 'object') return isRecord(value)
  return typeof value === type
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
