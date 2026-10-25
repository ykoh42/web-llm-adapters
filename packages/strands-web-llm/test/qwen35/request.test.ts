import {
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { qwen35Request } from '../../src/qwen35/request.js'
import { formatCall } from '../../../../shared/web-llm/qwen35/tools.js'

const tool = {
  name: 'lookup',
  description: 'Return the code for a city.',
  inputSchema: {
    type: 'object' as const,
    properties: { city: { type: 'string' as const } },
    required: ['city'],
  },
}

it('matches the Qwen3.5 system, thinking, tool-call, and tool-result formats', () => {
  const request = qwen35Request(
    { modelId: 'Qwen3.5-4B-q4f16_1-MLC' },
    [
      new Message({ role: 'user', content: [new TextBlock('Find the code for Seoul.')] }),
      new Message({
        role: 'assistant',
        content: [
          new ReasoningBlock({ text: ' Check the city.\n ' }),
          new TextBlock(' \nWorking. \n '),
          new ToolUseBlock({ toolUseId: 'lookup-1', name: 'lookup', input: { city: 'Seoul' } }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'lookup-1',
            status: 'success',
            content: [new TextBlock('oak-42')],
          }),
          new ToolResultBlock({
            toolUseId: 'lookup-2',
            status: 'success',
            content: [new TextBlock('pine-17')],
          }),
        ],
      }),
    ],
    { systemPrompt: 'Be concise.', toolSpecs: [tool] },
  )

  expect(request.messages[0]).toEqual({
    role: 'system',
    content:
      '# Tools\n\nYou have access to the following functions:\n\n<tools>\n' +
      '{"type": "function", "function": {"name": "lookup", "description": "Return the code for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}' +
      '\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n' +
      '<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n' +
      '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n' +
      '<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n' +
      '- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n' +
      '- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nBe concise.',
  })
  expect(request.messages.find((message) => message.role === 'assistant')?.content).toBe(
    '<think>\nCheck the city.\n</think>\n\nWorking.\n\n<tool_call>\n<function=lookup>\n' +
      '<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>',
  )
  expect(request.messages.at(-1)?.content).toBe(
    '<tool_response>\noak-42\n</tool_response>\n<tool_response>\npine-17\n</tool_response>',
  )
})

it('keeps punctuation, escapes, and Unicode inside tool schema strings', () => {
  const request = qwen35Request(
    { modelId: 'Qwen3.5-0.8B-q4f16_1-MLC' },
    [new Message({ role: 'user', content: [new TextBlock('Find the code.')] })],
    {
      toolSpecs: [
        {
          name: 'lookup',
          description: 'Read "x:y,z" from C:\\data\\codes.json.\n서울',
          inputSchema: {
            type: 'object',
            properties: { city: { type: 'string', enum: ['Seoul, KR', '서울:강남'] } },
            required: ['city'],
          },
        },
      ],
    },
  )
  const system = request.messages[0]?.content
  expect(typeof system).toBe('string')
  expect(system).toContain(
    String.raw`{"type": "function", "function": {"name": "lookup", "description": "Read \"x:y,z\" from C:\\data\\codes.json.\n서울", "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Seoul, KR", "서울:강남"]}}, "required": ["city"]}}}`,
  )
})

it('matches the template JSON for nested tool arguments in history', () => {
  const request = qwen35Request(
    { modelId: 'Qwen3.5-4B-q4f16_1-MLC' },
    [
      new Message({
        role: 'assistant',
        content: [
          new ToolUseBlock({
            toolUseId: 'lookup-1',
            name: 'lookup',
            input: { filters: { tags: ['one', 'two'], city: 'Seoul, KR' } },
          }),
        ],
      }),
    ],
    {},
  )
  expect(request.messages.at(-1)?.content).toContain(
    '<parameter=filters>\n{"tags": ["one", "two"], "city": "Seoul, KR"}\n</parameter>',
  )
})

it('drops thoughts from an older Qwen3.5 turn', () => {
  const request = qwen35Request(
    { modelId: 'Qwen3.5-4B-q4f16_1-MLC' },
    [
      new Message({ role: 'user', content: [new TextBlock('Old question.')] }),
      new Message({
        role: 'assistant',
        content: [new ReasoningBlock({ text: 'Old thought.' }), new TextBlock('Old answer.')],
      }),
      new Message({ role: 'user', content: [new TextBlock('New question.')] }),
    ],
    {},
  )

  expect(request.messages[1]?.content).toBe('Old answer.')
})

it('keeps the required empty thinking block in the current Qwen3.5 turn', () => {
  const request = qwen35Request(
    { modelId: 'Qwen3.5-4B-q4f16_1-MLC' },
    [
      new Message({ role: 'user', content: [new TextBlock('Answer plainly.')] }),
      new Message({ role: 'assistant', content: [new TextBlock('Plain answer.')] }),
    ],
    {},
  )

  expect(request.messages.at(-1)?.content).toBe('<think>\n\n</think>\n\nPlain answer.')
})

it('rejects tool signatures that Qwen3.5 cannot preserve', () => {
  const messages = [
    new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({
          toolUseId: 'call-1',
          name: 'lookup',
          input: { city: 'Seoul' },
          reasoningSignature: 'opaque-signature',
        }),
      ],
    }),
  ]

  expect(() => qwen35Request({ modelId: 'Qwen3.5-4B-q4f16_1-MLC' }, messages, {})).toThrow(
    'tool calls with reasoning signatures',
  )
})

it('uses JSON serialization for optional and custom XML argument values', () => {
  const date = new Date('2026-10-20T00:00:00.000Z')
  const input = {
    missing: undefined,
    empty: '',
    zero: 0,
    disabled: false,
    nullable: null,
    date,
    custom: { toJSON: () => 'literal' },
    values: [undefined, 0, false, ''],
  }

  expect(formatCall('lookup', input)).toBe(
    '<tool_call>\n<function=lookup>\n' +
      '<parameter=empty>\n\n</parameter>\n' +
      '<parameter=zero>\n0\n</parameter>\n' +
      '<parameter=disabled>\nfalse\n</parameter>\n' +
      '<parameter=nullable>\nnull\n</parameter>\n' +
      '<parameter=date>\n2026-10-20T00:00:00.000Z\n</parameter>\n' +
      '<parameter=custom>\nliteral\n</parameter>\n' +
      '<parameter=values>\n[null, 0, false, ""]\n</parameter>\n' +
      '</function>\n</tool_call>',
  )
  expect(Object.hasOwn(input, 'missing')).toBe(true)
  expect(input.date).toBe(date)
})

it('writes JSON-serializable strings without adding quotes to XML values', () => {
  const date = new Date('2026-10-20T00:00:00.000Z')

  expect(formatCall('lookup', { date, custom: { toJSON: () => 'literal' } })).toBe(
    '<tool_call>\n<function=lookup>\n' +
      '<parameter=date>\n2026-10-20T00:00:00.000Z\n</parameter>\n' +
      '<parameter=custom>\nliteral\n</parameter>\n' +
      '</function>\n</tool_call>',
  )
})

it.each([null, [], 42, 'literal', { toJSON: () => [] }])(
  'rejects JSON tool arguments that are not objects: %j',
  (input) => {
    expect(() => formatCall('lookup', input)).toThrow('arguments must be an object')
  },
)

it.each(['<parameter=x>', '</parameter>'])(
  'rejects an ambiguous raw string history value containing %s',
  (text) => {
    expect(() => formatCall('lookup', { text })).toThrow('cannot preserve parameter tags')
  },
)

it.each(['a>b', '>code', 'code>'])(
  'rejects a history argument name that would close its XML header: %s',
  (name) => {
    expect(() => formatCall('lookup', { [name]: 42 })).toThrow('parameter names')
  },
)
