import { expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { qwenToolResponseFormat } from '../../../../shared/web-llm/qwen/constraints.js'
import { xmlToolTag } from '../../../../shared/web-llm/qwen35/constraints.js'
import type { JSONSchema, ToolSpec } from '../../../../shared/web-llm/types.js'
import { checkRequest, type WebLLMRequest } from '../../../../shared/web-llm/request.js'

const qwen = 'Qwen3-4B-q4f16_1-MLC'
const xml = 'Qwen3.5-4B-q4f16_1-MLC'
const tool: ToolSpec = {
  name: 'lookup',
  description: 'Look up a value.',
  inputSchema: {
    type: 'object',
    properties: {
      code: { type: 'string', enum: ['001', '010'] },
      active: { type: 'boolean' },
    },
    required: ['code'],
    additionalProperties: false,
  },
}
const request = { constrainedToolCalls: true, extra_body: { enable_thinking: true } }
const modelTable = readFileSync(new URL('../../../../README.md', import.meta.url), 'utf8')
  .split('## Models\n')[1]
  ?.split('\n<details>')[0]
const documentedQwenModels = [...(modelTable?.matchAll(/`(Qwen[^`]+)`/g) ?? [])].flatMap((match) =>
  match[1] ? [match[1]] : [],
)

it.each([null, 'true', 1, {}, []])('rejects a non-boolean constraints option: %j', (value) => {
  expect(() => checkRequest({ constrainedToolCalls: value } as unknown as WebLLMRequest)).toThrow(
    'must be a boolean',
  )
})

it('supports constraints for every documented Qwen model', () => {
  expect(documentedQwenModels.length).toBeGreaterThan(0)
  for (const modelId of documentedQwenModels)
    expect(qwenToolResponseFormat(modelId, request, [tool])).toMatchObject({
      type: 'structural_tag',
    })
})

it.each([qwen, xml])('%s keeps constraints opt-in and respects disabled tools', (modelId) => {
  expect(qwenToolResponseFormat(modelId, {}, [tool])).toBeUndefined()
  expect(qwenToolResponseFormat(modelId, { constrainedToolCalls: false }, [tool])).toBeUndefined()
  expect(qwenToolResponseFormat(modelId, request, [], { any: {} })).toBeUndefined()
  expect(qwenToolResponseFormat(modelId, request, [tool], 'none')).toBeUndefined()
})

it('leaves literal thinking markers as text on a non-thinking model', () => {
  const format = qwenToolResponseFormat('Qwen2.5-0.5B-Instruct-q4f16_1-MLC', request, [tool])
  expect(format).toMatchObject({
    structural_tag: { format: { triggers: ['<tool_call>'], at_least_one: false } },
  })
  expect(JSON.stringify(format)).not.toContain('<think>')
})

it.each([qwen, xml])(
  '%s requires a tool tag rather than counting thinking as a call',
  (modelId) => {
    const result = qwenToolResponseFormat(modelId, request, [tool], { any: {} })
    expect(result).toMatchObject({
      structural_tag: {
        format: {
          type: 'sequence',
          elements: [
            {
              type: 'tag',
              end: '<tool_call>',
              content: { triggers: expect.arrayContaining(['<think>']) },
            },
            { type: 'or', elements: [{ begin: expect.stringContaining('lookup') }] },
            {
              type: 'triggered_tags',
              at_least_one: false,
              stop_after_first: false,
            },
          ],
        },
      },
    })
  },
)

it.each([qwen, xml])('%s keeps auto selection optional and restricts a named choice', (modelId) => {
  const tools = [tool, { ...tool, name: 'other' }]
  const auto = qwenToolResponseFormat(modelId, request, tools, { auto: {} })
  expect(auto).toMatchObject({
    structural_tag: { format: { type: 'triggered_tags', at_least_one: false } },
  })
  const named = qwenToolResponseFormat(modelId, request, tools, { tool: { name: 'lookup' } })
  expect(JSON.stringify(named)).toContain('lookup')
  expect(JSON.stringify(named)).not.toContain('other')
  expect(tools).toEqual([tool, { ...tool, name: 'other' }])
})

it('keeps spaces in Qwen3.5 constrained tool names', () => {
  const spaced = { ...tool, name: 'read file' }

  expect(xmlToolTag(spaced)).toMatchObject({ begin: '<tool_call>\n<function=read file>\n' })
  expect(
    JSON.stringify(qwenToolResponseFormat(xml, request, [spaced], { tool: { name: spaced.name } })),
  ).toContain('<function=read file>')
})

it('keeps explicit text output compatible with constrained tool calls', () => {
  const response_format = { type: 'text' as const }
  expect(qwenToolResponseFormat(qwen, { ...request, response_format }, [tool])).toMatchObject({
    type: 'structural_tag',
  })
})

it.each([{ type: 'json_object' as const }, { type: 'grammar' as const, grammar: 'root ::= "ok"' }])(
  'rejects conflicting structured response formats: %j',
  (response_format) => {
    expect(() => qwenToolResponseFormat(qwen, { ...request, response_format }, [tool])).toThrow(
      'cannot be combined',
    )
  },
)

it('handles a prefilled thought without opening it again and honors disabled thinking', () => {
  const choice = { any: {} }
  const prefilled = qwenToolResponseFormat(qwen, request, [tool], choice, true)
  expect(prefilled).toMatchObject({
    structural_tag: {
      format: {
        elements: [
          expect.objectContaining({ type: 'tag', begin: '', end: '</think>' }),
          { type: 'sequence' },
        ],
      },
    },
  })
  const disabled = qwenToolResponseFormat(
    qwen,
    { ...request, extra_body: { enable_thinking: false } },
    [tool],
    choice,
    true,
  )
  expect(disabled).toMatchObject({
    structural_tag: { format: { elements: [{ end: '<tool_call>' }, {}, {}] } },
  })
})

it.each([false, true])(
  'guards legacy stops inside optional thinking, prefilled: %s',
  (prefilled) => {
    const result = qwenToolResponseFormat(xml, request, [tool], { auto: {} }, prefilled)
    const guarded = {
      type: 'triggered_tags',
      triggers: ['<|im_end|', '<|endoftext|'],
      at_least_one: false,
      stop_after_first: false,
    }
    const thoughts = {
      type: 'triggered_tags',
      at_least_one: false,
      tags: expect.arrayContaining([
        expect.objectContaining({ begin: '<think>', content: expect.objectContaining(guarded) }),
      ]),
    }
    expect(result).toMatchObject({
      structural_tag: {
        format: prefilled
          ? { elements: [{ begin: '', content: guarded, end: '</think>' }, thoughts] }
          : thoughts,
      },
    })
  },
)

it('keeps XML string enum values raw and preserves JSON containers', () => {
  const nested = { type: 'array' as const, items: { type: 'string' as const, enum: ['a', 'b'] } }
  const schema = {
    ...tool.inputSchema!,
    properties: { ...tool.inputSchema!.properties, items: nested },
  }
  const saved = structuredClone(schema)
  const result = xmlToolTag({ ...tool, inputSchema: schema })
  expect(result).toMatchObject({
    content: {
      elements: [
        { content: { elements: [{ type: 'grammar', grammar: 'root ::= "001"' }, {}] } },
        { type: 'or' },
        { elements: [{ content: { type: 'json_schema', json_schema: nested } }, {}] },
      ],
    },
  })
  expect(schema).toEqual(saved)
})

it.each([{ enum: ['001', '010'] }, { const: '001' }])(
  'infers Qwen3.5 raw string formats from %j without a type',
  (value) => {
    const inputSchema = { ...tool.inputSchema, properties: { code: value } }
    expect(xmlToolTag({ ...tool, inputSchema })).toBeDefined()
  },
)

it.each([
  {
    anyOf: [
      {
        type: 'object',
        properties: { code: { type: 'string' } },
        required: ['code'],
        additionalProperties: false,
      },
      { type: 'null' },
    ],
  },
  { oneOf: [{ type: 'array', items: { type: 'string' } }, { type: 'null' }] },
  { anyOf: [{ type: 'number' }, { type: 'boolean' }] },
] satisfies JSONSchema[])(
  'passes type-less JSON unions to WebLLM as JSON Schema: %j',
  (payload) => {
    const inputSchema: JSONSchema = {
      ...tool.inputSchema,
      properties: { payload },
      required: ['payload'],
    }

    expect(xmlToolTag({ ...tool, inputSchema })).toMatchObject({
      content: { elements: [{ content: { type: 'json_schema', json_schema: payload } }] },
    })
  },
)

it('passes a type-less numeric enum to WebLLM as JSON Schema', () => {
  const code: JSONSchema = { enum: [7, 9] }
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { code },
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toMatchObject({
    content: { elements: [{ content: { type: 'json_schema', json_schema: code } }] },
  })
})

it.each([{ const: false }, { const: null }, { enum: [{ code: 'A7' }] }])(
  'passes a type-less JSON const or enum to WebLLM: %j',
  (code) => {
    const inputSchema: JSONSchema = {
      ...tool.inputSchema,
      properties: { code },
    }

    expect(xmlToolTag({ ...tool, inputSchema })).toMatchObject({
      content: { elements: [{ content: { type: 'json_schema', json_schema: code } }] },
    })
  },
)

it.each<JSONSchema>([
  { enum: [Number.NaN] },
  { const: undefined } as unknown as JSONSchema,
  { const: new Date(0) } as unknown as JSONSchema,
  { const: [undefined] } as unknown as JSONSchema,
])('rejects type-less values that JSON cannot preserve: %j', (code) => {
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { code },
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
})

it('rejects a cyclic type-less constant', () => {
  const value: Record<string, unknown> = {}
  value.self = value
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { code: { const: value } as JSONSchema },
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
})

it('rejects cyclic schema objects before passing them to WebLLM', () => {
  const recursive: JSONSchema = { type: 'object' }
  recursive.properties = { value: recursive }
  const inputSchema: JSONSchema = {
    type: 'object',
    properties: { payload: recursive },
    additionalProperties: false,
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
  expect(() => qwenToolResponseFormat(xml, request, [{ ...tool, inputSchema }])).toThrow(
    'cannot represent',
  )
})

it('rejects cycles inside JSON Schema annotation values', () => {
  const example: Record<string, unknown> = {}
  example.self = example
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { code: { type: 'string', default: example as JSONSchema['default'] } },
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
})

it('keeps shared acyclic schema objects usable', () => {
  const shared: JSONSchema = { type: 'string', enum: ['A7'] }
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { first: shared, second: shared },
    required: [],
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toBeDefined()
})

it('rejects an empty type-less enum even when a union is present', () => {
  const code: JSONSchema = {
    enum: [],
    anyOf: [{ type: 'integer' }, { type: 'null' }],
  }
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { code },
  }

  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
})

it.each([{ const: { $ref: 'literal' } }, { enum: [{ $ref: 'literal' }] }])(
  'keeps a literal reference key inside JSON values: %j',
  (code) => {
    const inputSchema: JSONSchema = {
      ...tool.inputSchema,
      properties: { code },
    }

    expect(xmlToolTag({ ...tool, inputSchema })).toMatchObject({
      content: { elements: [{ content: { type: 'json_schema', json_schema: code } }] },
    })
  },
)

it.each(['default', 'examples', 'enum', 'const'])(
  'finds schema references below a property named %s',
  (name) => {
    const inputSchema: JSONSchema = {
      ...tool.inputSchema,
      properties: {
        code: {
          type: 'object',
          properties: { [name]: { $ref: '#/definitions/value' } },
        },
      },
    }
    const nestedTool = { ...tool, inputSchema }

    expect(xmlToolTag(nestedTool)).toBeUndefined()
    expect(() => qwenToolResponseFormat(xml, request, [nestedTool])).toThrow('cannot represent')
    expect(qwenToolResponseFormat(xml, {}, [nestedTool])).toBeUndefined()
  },
)

it('guards raw strings at parser boundaries and reuses only the immutable grammar', () => {
  const schema: JSONSchema = {
    type: 'object',
    properties: { text: { type: 'string' } },
    required: ['text'],
    additionalProperties: false,
  }
  const first = xmlToolTag({ ...tool, inputSchema: schema })
  expect(first).toMatchObject({
    content: { elements: [{ content: { type: 'grammar', grammar: expect.any(String) } }] },
  })
  expect(xmlToolTag({ ...tool, inputSchema: structuredClone(schema) })).toEqual(first)
  expect(
    xmlToolTag({ ...tool, inputSchema: { type: 'object', additionalProperties: false } }),
  ).toMatchObject({
    content: { type: 'grammar', grammar: 'root ::= ""' },
  })
})

it.each<JSONSchema>([
  { type: 'string', const: '' },
  { type: 'string', const: '"quote"\\path\n서울' },
  { type: ['string', 'null'], enum: ['value', null] },
  { type: ['string', 'null'], enum: [' null '] },
])('preserves representable raw string choices: %j', (value) => {
  const inputSchema = { ...tool.inputSchema, properties: { code: value } }
  expect(xmlToolTag({ ...tool, inputSchema })).toBeDefined()
})

it.each<JSONSchema>([
  { type: 'string', const: 'value\r' },
  { type: 'string', enum: ['value</parameter>'] },
  { type: 'string', enum: ['<parameter=other>'] },
  { type: 'string', const: '<|im_start|>' },
  { type: 'string', enum: ['<|im_end|>'] },
  { type: ['string', 'null'], enum: [' null ', null] },
  { type: 'string', const: '001', enum: ['010'] },
  { type: 'string', enum: [] },
  { type: 'string', enum: [1] },
  { enum: ['001', 1] },
])('rejects raw string choices the protocol cannot preserve: %j', (value) => {
  const inputSchema = { ...tool.inputSchema, properties: { code: value } }
  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
})

it.each<JSONSchema>([
  { type: 'object', properties: {} },
  { type: 'object', additionalProperties: true },
  { ...tool.inputSchema, allOf: [] },
  { ...tool.inputSchema, required: ['missing'] },
  { ...tool.inputSchema, properties: { code: { $ref: '#/definitions/code' } } },
  { ...tool.inputSchema, properties: { code: { type: 'string', minLength: 1 } } },
  { ...tool.inputSchema, properties: { code: { type: ['string', 'number'] } } },
  { ...tool.inputSchema, properties: { code: false } },
])('rejects an XML schema it cannot constrain faithfully: %j', (inputSchema) => {
  expect(xmlToolTag({ ...tool, inputSchema })).toBeUndefined()
  expect(() => qwenToolResponseFormat(xml, request, [{ ...tool, inputSchema }])).toThrow(
    'cannot represent',
  )
  expect(qwenToolResponseFormat(xml, {}, [{ ...tool, inputSchema }])).toBeUndefined()
})

it.each<JSONSchema>([
  { type: ['string', 'null'] },
  { anyOf: [{ type: 'string' }, { type: 'null' }] },
])('rejects nullable strings the constrained XML parser cannot preserve: %j', (code) => {
  const inputSchema: JSONSchema = {
    ...tool.inputSchema,
    properties: { code },
  }
  const nullableString = { ...tool, inputSchema }

  expect(xmlToolTag(nullableString)).toBeUndefined()
  expect(() => qwenToolResponseFormat(xml, request, [nullableString])).toThrow('cannot represent')
})

it('rejects unregistered or unverified constrained tool selections', () => {
  expect(() =>
    qwenToolResponseFormat(qwen, request, [tool], { tool: { name: 'missing' } }),
  ).toThrow('not registered')
  expect(() => qwenToolResponseFormat('Qwen3.5-4B-q4f32_1-MLC', request, [tool])).toThrow(
    'not verified',
  )
})
