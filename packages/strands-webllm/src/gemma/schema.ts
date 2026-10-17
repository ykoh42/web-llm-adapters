const schemaMaps = ['properties', '$defs', 'definitions', 'patternProperties']
const schemaFields = ['items', 'additionalProperties', 'oneOf', 'anyOf', 'allOf', 'prefixItems']

/** Uppercase schema types without changing property names, defaults, or enum values. */
export function gemmaSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(gemmaSchema)
  if (typeof value !== 'object' || value === null) return value
  const fields: [string, unknown][] = []
  for (const [key, item] of Object.entries(value)) {
    if (key === 'type') {
      if (typeof item === 'string') {
        fields.push([key, item.toUpperCase()])
        continue
      }
      if (Array.isArray(item)) {
        fields.push([
          key,
          item.map((type: unknown) => (typeof type === 'string' ? type.toUpperCase() : type)),
        ])
        continue
      }
    }
    if (schemaMaps.includes(key) && typeof item === 'object' && item !== null) {
      fields.push([
        key,
        Object.fromEntries(Object.entries(item).map(([name, child]) => [name, gemmaSchema(child)])),
      ])
      continue
    }
    fields.push([key, schemaFields.includes(key) ? gemmaSchema(item) : item])
  }
  return Object.fromEntries(fields)
}
