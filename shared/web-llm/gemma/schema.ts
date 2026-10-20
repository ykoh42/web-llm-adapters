import type { JSONSchema } from '../types.js'

/** Render the fields and ordering used by Gemma's canonical declaration template. */
export function gemmaSchema(schema: JSONSchema, root = true): Record<string, unknown> {
  const type = schema.type
  const fields: Record<string, unknown> = {}
  if (!root && schema.description) fields.description = schema.description
  if (type === 'string' && schema.enum?.length) fields.enum = schema.enum
  if (type === 'array' && schema.items) fields.items = arrayItems(schema.items)
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
  const entries = Object.entries(values).sort(([a], [b]) => a.localeCompare(b))
  return Object.fromEntries(
    entries.map(([name, schema]) => [
      name,
      gemmaSchema(typeof schema === 'boolean' ? {} : schema, false),
    ]),
  )
}

function arrayItems(items: NonNullable<JSONSchema['items']>): unknown {
  if (Array.isArray(items)) return items.map(arrayItems)
  if (typeof items === 'boolean') return items
  const fields = Object.create(null) as Record<string, unknown>
  for (const [key, value] of Object.entries(items).sort(([a], [b]) => a.localeCompare(b))) {
    if (value == null) continue
    if (key === 'properties') {
      fields.properties = properties(items.properties ?? {})
    } else if (key === 'type' && items.type) {
      fields.type = Array.isArray(items.type) ? items.type.map(upperType) : upperType(items.type)
    } else fields[key] = value
  }
  return fields
}
