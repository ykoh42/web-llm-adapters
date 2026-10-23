import type { JSONSchema } from '../types.js'

/** Render the fields and ordering used by Gemma's canonical declaration template. */
export function gemmaSchema(schema: JSONSchema, root = true): Record<string, unknown> {
  const type = schema.type
  const fields: Record<string, unknown> = {}
  if (!root && schema.description) fields.description = schema.description
  if (type === 'string' && schema.enum?.length) fields.enum = schema.enum
  if (type === 'array' && schema.items !== undefined) fields.items = arrayItems(schema.items)
  if (!root && 'nullable' in schema && schema.nullable) fields.nullable = true
  if ((root || type === 'object') && schema.properties) {
    if (!root || Object.keys(schema.properties).length)
      fields.properties = properties(schema.properties)
  }
  if ((root || type === 'object') && schema.required?.length) fields.required = schema.required
  if (type) fields.type = Array.isArray(type) ? type.map(upperType) : upperType(type)
  return fields
}

function upperType(type: string) {
  return type.toUpperCase()
}

function properties(values: NonNullable<JSONSchema['properties']>) {
  return Object.fromEntries(
    Object.entries(values).map(([name, schema]) => [
      name,
      typeof schema === 'boolean' ? schema : gemmaSchema(schema, false),
    ]),
  )
}

function arrayItems(schema: NonNullable<JSONSchema['items']>): unknown {
  if (Array.isArray(schema)) return schema.map(arrayItems)
  if (typeof schema === 'boolean') return schema
  const fields = Object.create(null) as Record<string, unknown>
  for (const [key, value] of Object.entries(schema)) {
    if (value == null) continue
    if (key === 'properties') {
      fields.properties = properties(schema.properties ?? {})
    } else if (key === 'items') {
      fields.items = arrayItems(value as NonNullable<JSONSchema['items']>)
    } else if (key === 'type' && schema.type) {
      fields.type = Array.isArray(schema.type) ? schema.type.map(upperType) : upperType(schema.type)
    } else fields[key] = value
  }
  return fields
}
