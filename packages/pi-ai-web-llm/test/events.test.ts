import { describe, expect, it } from 'vitest'
import type { StreamChunk } from '../../../shared/web-llm/stream/chunks.js'
import { responseStream } from '../src/events.js'
import { collect, setup } from './helpers.js'

function chunk(values: Partial<StreamChunk>): StreamChunk {
  return { text: '', reasoning: '', calls: [], stop: undefined, usage: undefined, ...values }
}

function stream(...values: Partial<StreamChunk>[]) {
  const { model } = setup()
  return responseStream(model, () => ({
    chunks: (async function* () {
      for (const value of values) yield chunk(value)
    })(),
    toolNames: ['lookup'],
  }))
}

const call = (index: number, input: string, id = `call-${index}`) => ({
  index,
  id,
  name: 'lookup',
  input,
})

describe('Pi response events', () => {
  it('closes each text or thinking block before opening the next one', async () => {
    const response = stream(
      { reasoning: 'Plan' },
      { reasoning: ' carefully' },
      { text: 'First' },
      { reasoning: 'Reconsider' },
      { text: 'Final', stop: 'endTurn' },
    )
    const events = await collect(response)
    expect(events.map((event) => event.type)).toEqual([
      'start',
      'thinking_start',
      'thinking_delta',
      'thinking_delta',
      'thinking_end',
      'text_start',
      'text_delta',
      'text_end',
      'thinking_start',
      'thinking_delta',
      'thinking_end',
      'text_start',
      'text_delta',
      'text_end',
      'done',
    ])
    expect(
      events.filter((event) => event.type === 'thinking_end' || event.type === 'text_end'),
    ).toEqual([
      expect.objectContaining({ contentIndex: 0, content: 'Plan carefully' }),
      expect.objectContaining({ contentIndex: 1, content: 'First' }),
      expect.objectContaining({ contentIndex: 2, content: 'Reconsider' }),
      expect.objectContaining({ contentIndex: 3, content: 'Final' }),
    ])
    expect((await response.result()).content).toEqual([
      { type: 'thinking', thinking: 'Plan carefully' },
      { type: 'text', text: 'First' },
      { type: 'thinking', thinking: 'Reconsider' },
      { type: 'text', text: 'Final' },
    ])
  })

  it('publishes calls by index with their own original argument bytes', async () => {
    const response = stream({
      calls: [call(3, '{ "code": "B8" }'), call(1, '{"code":"A7"}')],
      stop: 'toolUse',
    })
    const events = await collect(response)
    expect(events.filter((event) => event.type === 'toolcall_delta')).toEqual([
      expect.objectContaining({ contentIndex: 0, delta: '{"code":"A7"}' }),
      expect.objectContaining({ contentIndex: 1, delta: '{ "code": "B8" }' }),
    ])
    expect((await response.result()).content).toEqual([
      { type: 'toolCall', id: 'call-1', name: 'lookup', arguments: { code: 'A7' } },
      { type: 'toolCall', id: 'call-3', name: 'lookup', arguments: { code: 'B8' } },
    ])
  })

  it.each([
    [call(0, '{}'), call(1, '{')],
    [call(0, '{}', 'same-id'), call(1, '{}', 'same-id')],
  ])('withholds all executable calls when any call is invalid %#', async (...calls) => {
    const response = stream({ calls, stop: 'toolUse' })
    const events = await collect(response)
    expect(events.map((event) => event.type)).toEqual(['start', 'error'])
    expect(await response.result()).toMatchObject({ stopReason: 'error', content: [] })
  })

  it('keeps final usage received after the stop chunk', async () => {
    const response = stream(
      { text: 'Done', stop: 'endTurn' },
      {
        usage: {
          prompt_tokens: 5,
          completion_tokens: 2,
          total_tokens: 7,
          extra: {
            e2e_latency_s: 0,
            prefill_tokens_per_s: 0,
            decode_tokens_per_s: 0,
            time_to_first_token_s: 0,
            time_per_output_token_s: 0,
          },
        },
      },
    )
    expect(await response.result()).toMatchObject({
      stopReason: 'stop',
      content: [{ type: 'text', text: 'Done' }],
      usage: { input: 5, output: 2, totalTokens: 7 },
    })
  })
})
