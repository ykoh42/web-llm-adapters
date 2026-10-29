import { ModelError, type ToolSpec } from '../types.js'

const jsonTypes = ['string', 'integer', 'nonInteger', 'boolean', 'null', 'array', 'object']
const annotations = new Set([
  'title',
  'description',
  '$comment',
  'default',
  'examples',
  'deprecated',
  'readOnly',
  'writeOnly',
])

interface SchemaContext {
  root: Record<string, unknown>
  name: string
  fields: Map<Record<string, unknown>, unknown[]>
  activeFields: Set<Record<string, unknown>>
  types: Map<Record<string, unknown>, string[] | undefined>
  activeTypes: Set<Record<string, unknown>>
  activeExactTypes: Set<Record<string, unknown>>
  unknownNegation: boolean
}

export interface ArgumentSchema {
  types: string[] | undefined
  unknownNegation: boolean
}

/** Reuse analysis within one request without caching mutable schemas between requests. */
export type ArgumentSchemaCache = WeakMap<ToolSpec, Map<string, ArgumentSchema>>

/** Only typed JSON containers may hide XML tags inside their quoted strings. */
export function isJsonArgument(tool: ToolSpec, name: string, cache?: ArgumentSchemaCache): boolean {
  const { types } = argumentSchema(tool, name, cache)
  return Boolean(
    types &&
    !types.includes('string') &&
    types.some((type) => type === 'object' || type === 'array'),
  )
}

export function argumentSchema(
  tool: ToolSpec,
  name: string,
  cache?: ArgumentSchemaCache,
): ArgumentSchema {
  const schemas = cache?.get(tool)
  const cached = schemas?.get(name)
  if (cached) return cached

  // A request-scoped cache keeps the next call current if a caller reuses and edits a schema.
  const context: SchemaContext | undefined = isRecord(tool.inputSchema)
    ? {
        root: tool.inputSchema,
        name,
        fields: new Map(),
        activeFields: new Set(),
        types: new Map(),
        activeTypes: new Set(),
        activeExactTypes: new Set(),
        unknownNegation: false,
      }
    : undefined
  const result = {
    types: context ? propertyTypes(context) : undefined,
    unknownNegation: context?.unknownNegation ?? false,
  }
  if (cache) {
    let schemas = cache.get(tool)
    if (!schemas) cache.set(tool, (schemas = new Map()))
    schemas.set(name, result)
  }
  return result
}

function propertyTypes(context: SchemaContext): string[] | undefined {
  const schemas = propertySchemas(context.root, context)
  const types = schemas
    .map((value) => readTypes(value, context))
    .filter((value): value is string[] => value !== undefined)
  if (!types.length) {
    if (context.unknownNegation)
      throw new ModelError('Qwen3.5 cannot determine a negated tool argument type.')
    return undefined
  }

  const [first, ...rest] = types
  const intersection = rest.reduce(intersectTypes, first!)
  if (!intersection.length) throw new ModelError('Qwen3.5 found incompatible tool argument types.')
  return intersection
}

function intersectTypes(types: string[], next: string[]) {
  return [...new Set(types.filter((type) => next.includes(type)))]
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
  if (context.activeFields.has(schema))
    throw new ModelError('Qwen3.5 found a recursive tool schema.')
  context.activeFields.add(schema)
  try {
    const schemas: unknown[] = []

    if (Object.hasOwn(schema, 'not') && schema.not !== undefined) {
      const excluded = excludedObject(schema, context, references)
      if (excluded === 'unknown') context.unknownNegation = true
      else if (excluded === 'object') schemas.push(false)
    }

    if (typeof schema.$ref === 'string') {
      if (references.has(schema.$ref))
        throw new ModelError('Qwen3.5 found a recursive tool schema.')
      const target = resolveReference(context.root, schema.$ref)
      if (typeof target !== 'boolean' && !isRecord(target))
        throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
      schemas.push(...propertySchemas(target, context, new Set([...references, schema.$ref])))
    }

    addProperties(schema, context, schemas)
    addPropertyBranches(schema, context, references, schemas)

    context.fields.set(schema, schemas)
    return schemas
  } finally {
    context.activeFields.delete(schema)
  }
}

function excludedObject(
  schema: Record<string, unknown>,
  context: SchemaContext,
  references: Set<string>,
): 'unknown' | 'object' | undefined {
  const excluded = exactTypes(schema.not, context, references)
  const types = isRecord(schema.not) ? schemaTypes(schema.not.type) : []
  // Object exclusions can depend on several arguments; do not invert one property.
  if (excluded === undefined && (!types.length || types.includes('object'))) return 'unknown'
  if (excluded?.includes('object')) return 'object'
  return undefined
}

function addProperties(
  schema: Record<string, unknown>,
  context: SchemaContext,
  schemas: unknown[],
) {
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
}

function addPropertyBranches(
  schema: Record<string, unknown>,
  context: SchemaContext,
  references: Set<string>,
  schemas: unknown[],
) {
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
}

function readTypes(
  value: unknown,
  context: SchemaContext,
  references = new Set<string>(),
): string[] | undefined {
  if (value === false) return []
  if (!isRecord(value)) return undefined
  if (context.types.has(value)) return context.types.get(value)
  if (context.activeTypes.has(value)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
  context.activeTypes.add(value)
  try {
    const constraints: string[][] = []
    const reference =
      typeof value.$ref === 'string'
        ? referencedTypes(value as { $ref: string }, context, references)
        : undefined
    if (reference) constraints.push(reference)

    const types = schemaTypes(value.type)
    if (types.length) constraints.push(types)
    if (Array.isArray(value.enum) && value.enum.length)
      constraints.push([...new Set(value.enum.map(valueType))])
    if (Object.hasOwn(value, 'const') && value.const !== undefined)
      constraints.push([valueType(value.const)])

    addTypeBranches(value, context, references, constraints)
    const negation =
      Object.hasOwn(value, 'not') && value.not !== undefined
        ? negatedTypes(value.not, context, references)
        : undefined
    if (negation) constraints.push(negation)

    if (!constraints.length) {
      context.types.set(value, undefined)
      return undefined
    }
    const [first, ...rest] = constraints
    const result = rest.reduce(intersectTypes, first!)
    context.types.set(value, result)
    return result
  } finally {
    context.activeTypes.delete(value)
  }
}

function referencedTypes(
  value: { $ref: string },
  context: SchemaContext,
  references: Set<string>,
): string[] | undefined {
  if (references.has(value.$ref)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
  const target = resolveReference(context.root, value.$ref)
  if (typeof target !== 'boolean' && !isRecord(target))
    throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
  return readTypes(target, context, new Set([...references, value.$ref]))
}

function negatedTypes(
  value: unknown,
  context: SchemaContext,
  references: Set<string>,
): string[] | undefined {
  const excluded = exactTypes(value, context, references)
  if (excluded === undefined) {
    context.unknownNegation = true
    return undefined
  }
  if (excluded.length) return jsonTypes.filter((type) => !excluded.includes(type))
  return undefined
}

function addTypeBranches(
  value: Record<string, unknown>,
  context: SchemaContext,
  references: Set<string>,
  constraints: string[][],
) {
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
}

function schemaTypes(value: unknown): string[] {
  const types = Array.isArray(value)
    ? value.filter((type): type is string => typeof type === 'string')
    : typeof value === 'string'
      ? [value]
      : []
  return types.flatMap((type) => (type === 'number' ? ['integer', 'nonInteger'] : [type]))
}

/** Only invert schemas whose entire meaning is a set of JSON types. */
function exactTypes(
  value: unknown,
  context: SchemaContext,
  references: Set<string>,
): string[] | undefined {
  if (value === false) return []
  if (value === true) return jsonTypes
  if (!isRecord(value)) return undefined
  if (context.activeExactTypes.has(value))
    throw new ModelError('Qwen3.5 found a recursive tool schema.')
  context.activeExactTypes.add(value)
  try {
    const keys = Object.keys(value).filter(
      (key) => !annotations.has(key) && value[key] !== undefined,
    )
    if (!keys.length) return jsonTypes
    if (keys.length !== 1) return undefined
    if (keys[0] === 'type') return schemaTypes(value.type)
    if (keys[0] === 'not') {
      const excluded = exactTypes(value.not, context, references)
      return excluded === undefined
        ? undefined
        : jsonTypes.filter((type) => !excluded.includes(type))
    }
    if (keys[0] === 'allOf' || keys[0] === 'anyOf') {
      const keyword = keys[0]
      return exactBranchTypes(value[keyword], keyword, context, references)
    }
    if (keys[0] !== '$ref') return undefined
    return exactReference(value, context, references)
  } finally {
    context.activeExactTypes.delete(value)
  }
}

function exactReference(
  value: Record<string, unknown>,
  context: SchemaContext,
  references: Set<string>,
): string[] | undefined {
  if (typeof value.$ref !== 'string' || !value.$ref.startsWith('#')) return undefined
  if (references.has(value.$ref)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
  const target = resolveReference(context.root, value.$ref)
  if (typeof target !== 'boolean' && !isRecord(target))
    throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
  return exactTypes(target, context, new Set([...references, value.$ref]))
}

function exactBranchTypes(
  alternatives: unknown,
  keyword: 'allOf' | 'anyOf',
  context: SchemaContext,
  references: Set<string>,
): string[] | undefined {
  if (!Array.isArray(alternatives) || !alternatives.length) return undefined
  const branches: string[][] = []
  for (const branch of alternatives) {
    const types = exactTypes(branch, context, references)
    if (types === undefined) return undefined
    branches.push(types)
  }
  return jsonTypes.filter((type) =>
    keyword === 'allOf'
      ? branches.every((types) => types.includes(type))
      : branches.some((types) => types.includes(type)),
  )
}

function valueType(value: unknown): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'nonInteger'
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
      continue
    }
    if (!isRecord(value) || !Object.hasOwn(value, key)) return undefined
    value = value[key]
  }
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
