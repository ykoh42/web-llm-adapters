import { expect, it, vi } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { chunk, chunks, setup } from './helpers.js'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const user = { role: 'user' as const, content: 'Look up A7.', timestamp: 0 }
const tools = [
  {
    name: 'lookup',
    description: 'Look up a code.',
    parameters: Type.Object({ code: Type.String() }),
  },
]
const context = normalizeContext({ messages: [user], tools })
const call = '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'

it('replays the required empty thinking block after a tool result', async () => {
  const { models, model, create } = setup([chunk({ content: '<think></think>' + call }, 'stop')], {
    modelId,
  })
  const first = await models.complete(model, context)
  const toolCall = first.content.find((part) => part.type === 'toolCall')!
  expect(toolCall).toMatchObject({ name: 'lookup', arguments: { code: 'A7' } })
  create.mockResolvedValueOnce(chunks(chunk({ content: '42' }, 'stop')))
  await models.complete(
    model,
    normalizeContext({
      tools,
      messages: [
        user,
        first,
        {
          role: 'toolResult',
          toolCallId: toolCall.id,
          toolName: 'lookup',
          content: [{ type: 'text', text: '42' }],
          isError: false,
          timestamp: 0,
        },
      ],
    }),
  )
  const assistant = create.mock.calls[1]![0].messages.find(
    (message) => message.role === 'assistant',
  )
  expect(assistant?.content).toMatch(/^<think>\n\n<\/think>\n\n<tool_call>/)
})

it.each([
  {
    body: '<tool_call><function=lookup><parameter=code>A7</parameter></function>',
    finish: 'stop',
    reason: 'error',
  },
  { body: '<tool_call><function=lookup><parameter=code>A7', finish: 'length', reason: 'length' },
  { body: '<tool_call><function=lookup><parameter=code>A7', finish: null, reason: 'error' },
] as const)('keeps tool syntax out of text on $finish', async ({ body, finish, reason }) => {
  const raw = 'Before<think>Plan</think>' + body
  const observed = vi.fn()
  const { models, model } = setup([chunk({ content: raw }, finish)], { modelId, onChunk: observed })
  const stream = models.stream(model, context)
  let text = ''
  for await (const event of stream) if (event.type === 'text_delta') text += event.delta
  const result = await stream.result()
  expect(result.stopReason).toBe(reason)
  expect(text).toBe('Before')
  expect(result.content).toEqual([
    { type: 'text', text: 'Before' },
    { type: 'thinking', thinking: 'Plan' },
  ])
  expect(observed.mock.calls.map(([item]) => item.choices[0]?.delta.content ?? '').join('')).toBe(
    raw,
  )
})

it('keeps literal tool syntax when tool parsing is not active', async () => {
  const text = '<tool_call><function=example>literal code'
  const { models, model } = setup([chunk({ content: text }, 'stop')], { modelId })
  const result = await models.complete(model, normalizeContext({ messages: [user] }))
  expect(result.stopReason).toBe('stop')
  expect(result.content).toEqual([{ type: 'text', text }])
})
