import { describe, expect, it } from 'vitest'
import { Type, type AssistantMessage, type Message } from '@earendil-works/pi-ai'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { formatContext } from '../src/messages.js'
import { setup } from './helpers.js'

function assistant(content: AssistantMessage['content']): AssistantMessage {
  return {
    role: 'assistant',
    api: 'webllm',
    provider: 'webllm',
    model: 'loaded-model',
    content,
    stopReason: 'toolUse',
    timestamp: 0,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  }
}
const toolCall = {
  type: 'toolCall' as const,
  id: 'call-1',
  name: 'lookup',
  arguments: { code: 'A7' },
}
const result: Message = {
  role: 'toolResult',
  toolCallId: 'call-1',
  toolName: 'lookup',
  content: [{ type: 'text', text: '42' }],
  isError: false,
  timestamp: 0,
}

describe('Pi transcript conversion', () => {
  it('preserves system section and tool order after updates and removals', () => {
    const { model } = setup()
    const tool = (name: string, description = name) => ({
      name,
      description,
      parameters: Type.Object({}),
    })
    const context = normalizeContext({
      systemPrompt: 'Base',
      tools: [tool('lookup'), tool('removed'), tool('search')],
      messages: [
        {
          role: 'system',
          content: 'Extra',
          sections: { first: 'Old', removed: 'Remove me', last: 'Last' },
          timestamp: 1,
        },
        { role: 'user', content: 'Hello', timestamp: 2 },
        {
          role: 'system',
          content: '',
          sections: { first: 'First', removed: null, added: 'Added' },
          toolsRemoved: [{ name: 'removed' }],
          toolsAdded: [tool('lookup', 'Updated lookup'), tool('added')],
          timestamp: 3,
        },
      ],
    })
    const before = structuredClone(context)
    expect(formatContext(context, model)).toEqual({
      history: [
        { role: 'system', content: 'Base\n\nExtra\n\nFirst\n\nLast\n\nAdded' },
        { role: 'user', content: 'Hello' },
      ],
      tools: [
        { name: 'lookup', description: 'Updated lookup' },
        { name: 'search', description: 'search' },
        { name: 'added', description: 'added' },
      ].map((declaration) => ({
        ...declaration,
        inputSchema: { type: 'object', properties: {} },
      })),
    })
    expect(context).toEqual(before)
  })

  it.each([
    { messages: [], history: [] },
    {
      messages: [{ role: 'system', content: '', timestamp: 0 }],
      history: [{ role: 'system', content: '' }],
    },
    {
      messages: [{ role: 'user', content: '', timestamp: 0 }],
      history: [{ role: 'user', content: '' }],
    },
  ] satisfies { messages: Message[]; history: { role: string; content: string }[] }[])(
    'preserves empty transcript and message content %#',
    ({ messages, history }) => {
      const { model } = setup()
      expect(formatContext(normalizeContext({ messages }), model)).toEqual({ history, tools: [] })
    },
  )

  it('preserves mixed assistant blocks and call order without mutating the transcript', () => {
    const { model } = setup()
    const secondCall = { ...toolCall, id: 'call-2', arguments: { code: 'B8' } }
    const context = normalizeContext({
      messages: [
        assistant([
          { type: 'text', text: 'First' },
          { type: 'thinking', thinking: 'Plan' },
          toolCall,
          { type: 'text', text: 'Last' },
          { type: 'thinking', thinking: 'Again' },
          secondCall,
        ]),
        result,
        { ...result, toolCallId: 'call-2' },
      ],
    })
    const before = structuredClone(context)
    expect(formatContext(context, model).history[0]).toEqual({
      role: 'assistant',
      content: 'FirstLast',
      reasoning: 'PlanAgain',
      tool_calls: [
        {
          id: 'call-1',
          type: 'function',
          function: { name: 'lookup', arguments: '{"code":"A7"}' },
        },
        {
          id: 'call-2',
          type: 'function',
          function: { name: 'lookup', arguments: '{"code":"B8"}' },
        },
      ],
    })
    expect(context).toEqual(before)
  })

  it('keeps same-model thinking and omits foreign or redacted thinking', () => {
    const { model } = setup()
    const previous = assistant([
      { type: 'thinking', thinking: 'Reason' },
      { type: 'thinking', thinking: 'Secret', redacted: true },
      { type: 'text', text: 'Answer' },
    ])
    const format = (message: Message) =>
      formatContext(normalizeContext({ messages: [message] }), model).history[0]
    expect(format(previous)).toMatchObject({ reasoning: 'Reason', content: 'Answer' })
    expect(format({ ...previous, provider: 'other' })).not.toHaveProperty('reasoning')
    expect(format({ ...previous, stopReason: 'aborted' })).toBeUndefined()
  })

  it('preserves tool errors and sends tool images after all tool results', () => {
    const { model } = setup(undefined, { imageInputs: true })
    const input = normalizeContext({
      messages: [
        assistant([toolCall, { ...toolCall, id: 'call-2' }]),
        {
          ...result,
          isError: true,
          content: [
            { type: 'text', text: '4' },
            { type: 'image', mimeType: 'image/png', data: 'AQ==' },
            { type: 'text', text: '2' },
          ],
        },
        {
          ...result,
          toolCallId: 'call-2',
          content: [
            { type: 'image', mimeType: 'image/png', data: 'Ag==' },
            { type: 'text', text: 'Ready' },
          ],
        },
      ],
    })
    const before = structuredClone(input)
    expect(formatContext(input, model).history).toEqual([
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'call-1',
            type: 'function',
            function: { name: 'lookup', arguments: '{"code":"A7"}' },
          },
          {
            id: 'call-2',
            type: 'function',
            function: { name: 'lookup', arguments: '{"code":"A7"}' },
          },
        ],
      },
      { role: 'tool', tool_call_id: 'call-1', content: '[ERROR] 42' },
      { role: 'tool', tool_call_id: 'call-2', content: 'Ready' },
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AQ==' } },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,Ag==' } },
        ],
      },
    ])
    expect(input).toEqual(before)
  })

  it('preserves the error prefix for an empty tool error', () => {
    const { model } = setup()
    const error = { ...result, content: [], isError: true }
    const context = normalizeContext({ messages: [assistant([toolCall]), error] })

    expect(formatContext(context, model).history.at(-1)).toEqual({
      role: 'tool',
      tool_call_id: 'call-1',
      content: '[ERROR] ',
    })
  })

  it.each([
    [assistant([toolCall])],
    [assistant([toolCall]), { role: 'user', content: 'Next', timestamp: 0 }],
    [assistant([toolCall]), { ...result, toolName: 'other' }],
    [assistant([toolCall]), result, result],
    [assistant([toolCall]), result, assistant([toolCall]), result],
    [result],
  ] satisfies Message[][])(
    'rejects incomplete, mismatched or duplicated tool history %#',
    (...messages) => {
      const { model } = setup()
      expect(() => formatContext(normalizeContext({ messages }), model)).toThrow()
    },
  )

  it.each([
    { ...toolCall, argumentsError: 'Malformed JSON' },
    { ...toolCall, namespace: 'remote' },
  ])('rejects unreplayable calls %#', (call) => {
    const { model } = setup()
    expect(() =>
      formatContext(normalizeContext({ messages: [assistant([call]), result] }), model),
    ).toThrow('cannot replay')
  })

  it('falls back to ordinary tool declarations for preferred strict schemas', () => {
    const { model } = setup()
    const context = normalizeContext({
      messages: [],
      tools: [
        {
          name: 'lookup',
          description: 'Lookup',
          parameters: Type.Object({}),
          constrainedSampling: { type: 'json_schema', strict: 'prefer' },
        },
      ],
    })
    expect(formatContext(context, model).tools).toEqual([
      { name: 'lookup', description: 'Lookup', inputSchema: { type: 'object', properties: {} } },
    ])
  })
})
