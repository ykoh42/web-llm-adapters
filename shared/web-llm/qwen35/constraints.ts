import type { JSONSchema, ToolSpec } from '../types.js'
import { xmlStringFormat } from './grammar.js'

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

/** Raw XML strings must stay raw; the bundled XML schema compiler quotes enums. */
export function xmlToolTag(tool: ToolSpec) {
  const schema = tool.inputSchema
  if (
    !tool.name ||
    /[<>\s]/u.test(tool.name) ||
    !schema ||
    schema.type !== 'object' ||
    schema.additionalProperties !== false ||
    Object.keys(schema).some((key) => !objectKeys.has(key))
  )
    return
  const properties = schema.properties ?? {}
  if (schema.required?.some((name) => !Object.hasOwn(properties, name))) return
  const elements = []
  for (const [name, value] of Object.entries(properties)) {
    if (!name || /[<>\r\n]/u.test(name) || typeof value === 'boolean' || hasReference(value)) return
    const content = parameterFormat(value)
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

function parameterFormat(schema: JSONSchema) {
  const types = Array.isArray(schema.type) ? schema.type : [schema.type]
  if (!types.includes('string')) {
    if (!schema.type) return
    return { type: 'json_schema', json_schema: schema }
  }
  if (
    types.some((type) => type !== 'string' && type !== 'null') ||
    Object.keys(schema).some((key) => !stringKeys.has(key))
  )
    return
  const values = Object.hasOwn(schema, 'const') ? [schema.const] : schema.enum
  if (!values) return xmlStringFormat()
  if (
    !values.length ||
    (Object.hasOwn(schema, 'const') && schema.enum && !schema.enum.includes(schema.const!)) ||
    values.some(
      (value) =>
        (typeof value !== 'string' && !(value === null && types.includes('null'))) ||
        (typeof value === 'string' &&
          (/<parameter=|<\/parameter>|<\|im_end\|>|<\|endoftext\|>/u.test(value) ||
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

function hasReference(value: unknown, seen = new WeakSet<object>()): boolean {
  if (!value || typeof value !== 'object') return false
  if (seen.has(value)) return false
  seen.add(value)
  return (
    Object.hasOwn(value, '$ref') || Object.values(value).some((entry) => hasReference(entry, seen))
  )
}
