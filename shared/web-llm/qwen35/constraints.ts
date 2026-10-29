import type { JSONSchema, ToolSpec } from '../types.js'
import { messageBoundaryTokens, xmlStringFormat } from './grammar.js'
import { argumentSchema, type ArgumentSchemaCache } from './schema.js'

export function canUseXmlFunctionName(name: unknown): name is string {
  return typeof name === 'string' && name.length > 0 && !/[<>]/u.test(name)
}

const annotations = [
  'title',
  'description',
  '$comment',
  'default',
  'examples',
  'deprecated',
  'readOnly',
  'writeOnly',
]
const objectKeys = new Set([
  ...annotations,
  '$schema',
  'type',
  'properties',
  'required',
  'additionalProperties',
])
const stringKeys = new Set([...annotations, 'type', 'enum', 'const'])
const jsonValueTypes = new Set(['integer', 'nonInteger', 'boolean', 'null', 'array', 'object'])
const jsonValueKeywords = new Set(['default', 'examples', 'enum', 'const'])
const schemaMapKeywords = new Set([
  'properties',
  'patternProperties',
  'definitions',
  '$defs',
  'dependencies',
  'dependentSchemas',
])

/** Raw XML strings must stay raw; the bundled XML schema compiler quotes enums. */
export function xmlToolTag(tool: ToolSpec) {
  const schema = tool.inputSchema
  if (
    !canUseXmlFunctionName(tool.name) ||
    !schema ||
    hasCycle(schema) ||
    schema.type !== 'object' ||
    schema.additionalProperties !== false ||
    Object.keys(schema).some((key) => !objectKeys.has(key))
  )
    return
  const properties = schema.properties ?? {}
  if (schema.required?.some((name) => !Object.hasOwn(properties, name))) return
  const argumentSchemas: ArgumentSchemaCache = new WeakMap()
  const elements = []
  for (const [name, value] of Object.entries(properties)) {
    if (!name || /[<>\r\n]/u.test(name) || typeof value === 'boolean' || hasReference(value)) return
    const content = parameterFormat(value, tool, name, argumentSchemas)
    if (!content) return
    const parameter = {
      type: 'tag',
      begin: `<parameter=${name}>\n`,
      content,
      end: '\n</parameter>\n',
    }
    elements.push(
      schema.required?.includes(name)
        ? parameter
        : { type: 'or', elements: [parameter, { type: 'grammar', grammar: 'root ::= ""' }] },
    )
  }
  return {
    type: 'tag' as const,
    begin: `<tool_call>\n<function=${tool.name}>\n`,
    content: elements.length
      ? { type: 'sequence', elements }
      : { type: 'grammar', grammar: 'root ::= ""' },
    end: '</function>\n</tool_call>',
  }
}

function parameterFormat(
  schema: JSONSchema,
  tool: ToolSpec,
  name: string,
  cache: ArgumentSchemaCache,
) {
  if (Array.isArray(schema.enum) && !schema.enum.length) return
  const values = Object.hasOwn(schema, 'const') ? [schema.const] : schema.enum
  const types = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : []
  if (!types.length) return untypedParameterFormat(schema, tool, name, cache, values)
  if (!types.includes('string')) {
    if (!schema.type) return
    return { type: 'json_schema', json_schema: schema }
  }
  return stringParameterFormat(schema, types, values)
}

function untypedParameterFormat(
  schema: JSONSchema,
  tool: ToolSpec,
  name: string,
  cache: ArgumentSchemaCache,
  values: readonly unknown[] | undefined,
) {
  if (Array.isArray(values) && values.length) {
    if (!values.some((value) => typeof value === 'string'))
      return inferredJsonFormat(schema, tool, name, cache, values)
    return stringParameterFormat(
      schema,
      values.includes(null) ? ['string', 'null'] : ['string'],
      values,
    )
  }
  const hasTypeUnion =
    (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) ||
    (Array.isArray(schema.oneOf) && schema.oneOf.length > 0)
  if (!hasTypeUnion) return
  return inferredJsonFormat(schema, tool, name, cache)
}

function inferredJsonFormat(
  schema: JSONSchema,
  tool: ToolSpec,
  name: string,
  cache: ArgumentSchemaCache,
  values?: readonly unknown[],
) {
  const inferred = argumentSchema(tool, name, cache).types
  if (
    (values === undefined || isJsonValue(values)) &&
    inferred?.length &&
    inferred.every((type) => jsonValueTypes.has(type))
  )
    return { type: 'json_schema', json_schema: schema }
  return undefined
}

function stringParameterFormat(
  schema: JSONSchema,
  types: string[],
  values: readonly unknown[] | undefined,
) {
  if (
    types.some((type) => type !== 'string' && type !== 'null') ||
    Object.keys(schema).some((key) => !stringKeys.has(key))
  )
    return
  if (!values) return types.includes('null') ? undefined : xmlStringFormat()
  if (
    !values.length ||
    (Object.hasOwn(schema, 'const') && schema.enum && !schema.enum.includes(schema.const!)) ||
    values.some(
      (value) =>
        (typeof value !== 'string' && !(value === null && types.includes('null'))) ||
        (typeof value === 'string' &&
          (/<parameter=|<\/parameter>/u.test(value) ||
            messageBoundaryTokens.some((token) => value.includes(token)) ||
            value.endsWith('\r') ||
            (value.trim() === 'null' && values.includes(null)))),
    )
  )
    return
  return {
    type: 'or',
    elements: values.map((value) => ({
      type: 'grammar',
      grammar: `root ::= ${JSON.stringify(String(value))}`,
    })),
  }
}

function hasReference(value: unknown, seen = new WeakSet<object>(), schemaMap = false): boolean {
  if (!value || typeof value !== 'object') return false
  if (seen.has(value)) return false
  seen.add(value)
  if (schemaMap) return Object.values(value).some((schema) => hasReference(schema, seen))
  return (
    Object.hasOwn(value, '$ref') ||
    Object.entries(value).some(
      ([key, entry]) =>
        !jsonValueKeywords.has(key) && hasReference(entry, seen, schemaMapKeywords.has(key)),
    )
  )
}

function hasCycle(
  value: unknown,
  seen = new WeakSet<object>(),
  active = new WeakSet<object>(),
): boolean {
  if (!value || typeof value !== 'object') return false
  if (active.has(value)) return true
  if (seen.has(value)) return false
  seen.add(value)
  active.add(value)
  try {
    return Object.values(value).some((entry) => hasCycle(entry, seen, active))
  } finally {
    active.delete(value)
  }
}

function isJsonValue(value: unknown, active = new WeakSet<object>()): boolean {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (typeof value !== 'object' || active.has(value) || !hasJsonPrototype(value)) return false

  const array = Array.isArray(value)
  const keys = Reflect.ownKeys(value)
  if (
    (array &&
      (keys.length !== value.length + 1 ||
        Object.keys(value).length !== value.length ||
        !Array.from({ length: value.length }, (_, index) => Object.hasOwn(value, index)).every(
          Boolean,
        ))) ||
    (!array && keys.length !== Object.keys(value).length)
  )
    return false

  active.add(value)
  const valid = keys
    .filter((key) => key !== 'length')
    .every((key) => {
      if (typeof key !== 'string') return false
      const descriptor = Object.getOwnPropertyDescriptor(value, key)
      return Boolean(
        descriptor?.enumerable &&
        Object.hasOwn(descriptor, 'value') &&
        isJsonValue(descriptor.value, active),
      )
    })
  active.delete(value)
  return valid
}

function hasJsonPrototype(value: object) {
  const prototype = Object.getPrototypeOf(value)
  if (prototype === null) return true
  const parent = Object.getPrototypeOf(prototype)
  return Array.isArray(value)
    ? parent !== null && Object.getPrototypeOf(parent) === null
    : parent === null
}
