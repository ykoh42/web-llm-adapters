import type { JSONSchema7 } from 'json-schema'

/** A small schema with many paths to the same definition. */
export function sharedSchema(depth = 17) {
  const definitions: Record<string, JSONSchema7> = {
    level0: { properties: { code: { type: 'string' } }, required: ['code'] },
  }
  for (let level = 1; level <= depth; level++) {
    const $ref = '#/definitions/level' + (level - 1)
    definitions['level' + level] = { allOf: [{ $ref }, { $ref }] }
  }
  return {
    type: 'object',
    definitions,
    $ref: '#/definitions/level' + depth,
  } satisfies JSONSchema7
}
