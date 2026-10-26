import {
  ImageBlock,
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { qwenRequest } from '../../src/qwen/request.js'
import { ModelType, postInitAndCheckFieldsChatCompletion } from '@mlc-ai/web-llm'

it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  '%s keeps validation order when history has several invalid blocks',
  (modelId) => {
    const image = new ImageBlock({
      format: 'png',
      source: { url: 'https://example.com/image.png' },
    })
    const thought = new ReasoningBlock({ text: 'Plan.', signature: 'signed' })
    const call = new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: { value: NaN } })
    const signed = new ToolUseBlock({
      toolUseId: 'signed',
      name: 'lookup',
      input: {},
      reasoningSignature: 'signed',
    })
    for (const { content, error } of [
      { content: [image, call], error: 'imageBlock here' },
      { content: [image, thought, call], error: 'signed or redacted reasoning' },
      { content: [image, thought, signed], error: 'tool calls with reasoning signatures' },
    ])
      expect(() =>
        qwenRequest({ modelId }, [new Message({ role: 'assistant', content })], {}),
      ).toThrow(error)
  },
)

it('allows automatic Qwen3 1.7B chat when no tools are registered', () => {
  const modelId = 'Qwen3-1.7B-q4f16_1-MLC'
  const request = qwenRequest(
    { modelId },
    [new Message({ role: 'user', content: [new TextBlock('Hi')] })],
    { toolChoice: { auto: {} } },
  )

  expect(request).not.toHaveProperty('tools')
  expect(request).not.toHaveProperty('tool_choice')
  expect(() => postInitAndCheckFieldsChatCompletion(request, modelId, ModelType.LLM)).not.toThrow()
})

it.each([
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
])('%s keeps its system prompt', (modelId) => {
  const request = qwenRequest(
    { modelId },
    [new Message({ role: 'user', content: [new TextBlock('Hi')] })],
    {},
  )

  expect(request.messages[0]).toEqual({
    role: 'system',
    content: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
  })
})

it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  '%s adds no default system text to the tool protocol',
  (modelId) => {
    const request = qwenRequest(
      { modelId },
      [new Message({ role: 'user', content: [new TextBlock('Find the code.')] })],
      {
        toolSpecs: [
          {
            name: 'lookup',
            description: 'Return the code for a city.',
            inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
          },
        ],
      },
    )

    expect(request.messages[0]?.role).toBe('system')
    expect(request.messages[0]?.content).not.toContain('You are a helpful assistant.')
  },
)

it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
  '%s keeps thoughts for Harness context but drops them after a user question',
  (modelId) => {
    const messages = [
      new Message({ role: 'user', content: [new TextBlock('Find the code.')] }),
      new Message({
        role: 'assistant',
        content: [
          new ReasoningBlock({ text: 'Check the code.' }),
          new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: {} }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'call',
            status: 'success',
            content: [new TextBlock('42')],
          }),
          new TextBlock('Current task context.'),
        ],
      }),
    ]
    const original = JSON.stringify(messages)

    const context = qwenRequest({ modelId }, messages, { dynamicTrailingBlocks: 1 })
    const followup = qwenRequest({ modelId }, messages, {})

    expect(context.messages[1]?.content).toContain('<think>\nCheck the code.\n</think>')
    expect(context.messages.at(-1)?.content).toBe('Current task context.')
    expect(followup.messages[1]?.content).not.toContain('Check the code.')
    expect(JSON.stringify(messages)).toBe(original)
  },
)

it.each(['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC'])(
  '%s keeps compact Hermes JSON for the WebLLM artifact without changing strings',
  (modelId) => {
    const request = qwenRequest({ modelId }, [], {
      toolSpecs: [
        {
          name: 'lookup',
          description: 'Read "x:y,z".\n서울',
          inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
        },
      ],
    })
    expect(request.messages[0]?.content).toContain(
      String.raw`{"type":"function","function":{"name":"lookup","description":"Read \"x:y,z\".\n서울","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}`,
    )
  },
)

it.each(['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC'])(
  '%s uses the official tool-call format',
  (modelId) => {
    const input = { path: '/tmp/note.txt', offset: 1, limit: 1 }
    const request = qwenRequest(
      { modelId },
      [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'read-1', name: 'read', input })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'read-1',
              status: 'success',
              content: [new TextBlock('The code is oak-42.')],
            }),
          ],
        }),
      ],
      {},
    )
    const assistant = request.messages.find((message) => message.role === 'assistant')!
      .content as string
    const call = assistant.match(/<tool_call>\n([\s\S]*?)\n<\/tool_call>/)
    expect(call).not.toBeNull()
    expect(call![1]).toBe(
      '{"name": "read", "arguments": {"path":"/tmp/note.txt","offset":1,"limit":1}}',
    )
    expect(request.messages.at(-1)).toEqual({
      role: 'user',
      content: '<tool_response>\nThe code is oak-42.\n</tool_response>',
    })
  },
)

it('groups consecutive Qwen tool results in one user turn', () => {
  const request = qwenRequest(
    { modelId: 'Qwen3-4B-q4f16_1-MLC' },
    [
      new Message({ role: 'user', content: [new TextBlock('Read both notes.')] }),
      new Message({
        role: 'assistant',
        content: [
          new ToolUseBlock({ toolUseId: 'read-1', name: 'read', input: { path: 'one.txt' } }),
          new ToolUseBlock({ toolUseId: 'read-2', name: 'read', input: { path: 'two.txt' } }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'read-1',
            status: 'success',
            content: [new TextBlock('First result.')],
          }),
          new ToolResultBlock({
            toolUseId: 'read-2',
            status: 'success',
            content: [new TextBlock('Second result.')],
          }),
        ],
      }),
    ],
    {},
  )

  expect(request.messages.at(-1)).toEqual({
    role: 'user',
    content:
      '<tool_response>\nFirst result.\n</tool_response>\n<tool_response>\nSecond result.\n</tool_response>',
  })
})

it('keeps Qwen3 thoughts only in the current tool loop', () => {
  const tool = new ToolUseBlock({ toolUseId: 'lookup-1', name: 'lookup', input: { city: 'Seoul' } })
  const messages = [
    new Message({ role: 'user', content: [new TextBlock('Look up Seoul.')] }),
    new Message({
      role: 'assistant',
      content: [
        new ReasoningBlock({ text: '\n\n  Find the code.  \n\n' }),
        new TextBlock('\n\nWorking.  '),
        tool,
      ],
    }),
    new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'lookup-1',
          status: 'success',
          content: [new TextBlock('The code is oak-42.')],
        }),
      ],
    }),
  ]

  const request = qwenRequest({ modelId: 'Qwen3-4B-q4f16_1-MLC' }, messages, {})
  expect(request.messages[1]?.content).toBe(
    '<think>\n  Find the code.  \n</think>\n\nWorking.  \n<tool_call>\n{"name": "lookup", "arguments": {"city":"Seoul"}}\n</tool_call>',
  )

  const nextTurn = qwenRequest(
    { modelId: 'Qwen3-4B-q4f16_1-MLC' },
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
  expect(nextTurn.messages[1]?.content).toBe('Old answer.')
})

it('does not add an empty thinking block to a Qwen3 answer', () => {
  const request = qwenRequest(
    { modelId: 'Qwen3-4B-q4f16_1-MLC' },
    [
      new Message({ role: 'user', content: [new TextBlock('Answer plainly.')] }),
      new Message({ role: 'assistant', content: [new TextBlock('Plain answer.')] }),
    ],
    {},
  )

  expect(request.messages.at(-1)?.content).toBe('Plain answer.')
})

it('does not put Qwen2.5 reasoning blocks in the user-facing transcript', () => {
  const request = qwenRequest(
    { modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC' },
    [
      new Message({ role: 'user', content: [new TextBlock('Old question.')] }),
      new Message({
        role: 'assistant',
        content: [new ReasoningBlock({ text: 'Private thought.' }), new TextBlock('Old answer.')],
      }),
      new Message({ role: 'user', content: [new TextBlock('New question.')] }),
    ],
    {},
  )

  expect(request.messages.find((message) => message.role === 'assistant')?.content).toBe(
    'Old answer.',
  )
})

it('rejects tool signatures that Qwen cannot preserve', () => {
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

  expect(() => qwenRequest({ modelId: 'Qwen3-4B-q4f16_1-MLC' }, messages, {})).toThrow(
    'tool calls with reasoning signatures',
  )
})

it.each(['Qwen3-1.7B-q4f16_1-MLC', 'Qwen3-1.7B-q4f32_1-MLC'])(
  '%s preserves tool calls and their results in history',
  (modelId) => {
    const request = qwenRequest(
      { modelId },
      [
        new Message({ role: 'user', content: [new TextBlock('Look up A7.')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: { code: 'A7' } })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call',
              status: 'success',
              content: [new TextBlock('ORCHID-42')],
            }),
          ],
        }),
      ],
      {
        toolSpecs: [
          {
            name: 'lookup',
            description: 'Look up a code.',
            inputSchema: {
              type: 'object',
              properties: { code: { type: 'string' } },
              required: ['code'],
            },
          },
        ],
      },
    )
    expect(request.messages[0]?.content).toContain('"name":"lookup"')
    expect(request.messages[1]).toEqual({ role: 'user', content: 'Look up A7.' })
    const call = /<tool_call>\n([\s\S]*?)\n<\/tool_call>/.exec(String(request.messages[2]?.content))
    expect(JSON.parse(call?.[1] ?? 'null')).toEqual({ name: 'lookup', arguments: { code: 'A7' } })
    expect(request.messages[3]?.content).toContain('ORCHID-42')
    expect(() =>
      postInitAndCheckFieldsChatCompletion(request, modelId, ModelType.LLM),
    ).not.toThrow()
  },
)
