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
        assistant([toolCall]),
        {
          ...result,
          isError: true,
          content: [...result.content, { type: 'image', mimeType: 'image/png', data: 'AQ==' }],
        },
      ],
    })
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
        ],
      },
      { role: 'tool', tool_call_id: 'call-1', content: '[ERROR] 42' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AQ==' } }],
      },
    ])
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
