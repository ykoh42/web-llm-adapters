import { expect, it } from 'vitest'
import type { ModelStreamEvent } from '@strands-agents/sdk'
import { readStream } from '../../src/stream/read.js'
import type { StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'

it('keeps text after a tool call in its original order', async () => {
  const events = await collect(
    (async function* () {
      yield streamChunk({ text: 'Before' })
      yield streamChunk({
        calls: [{ index: 0, id: 'call-1', name: 'lookup', input: '{"value":' }],
      })
      yield streamChunk({ text: 'After' })
      yield streamChunk({
        calls: [{ index: 0, id: undefined, name: undefined, input: '1}' }],
        stop: 'toolUse',
      })
    })(),
  )

  const output = events.flatMap((event) => {
    if (event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart')
      return [`tool:${event.start.name}`]
    if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
      return [`text:${event.delta.text}`]
    return []
  })

  expect(output).toEqual(['text:Before', 'tool:lookup', 'text:After'])
})

it('keeps deferred text and suppresses tool calls when the source fails', async () => {
  const failure = new Error('generation failed')
  const events: ModelStreamEvent[] = []
  let thrown: unknown

  try {
    for await (const event of readStream(
      (async function* () {
        yield streamChunk({
          calls: [{ index: 0, id: 'call-1', name: 'lookup', input: '{}' }],
        })
        yield streamChunk({ text: 'Keep this text' })
        throw failure
      })(),
      undefined,
      undefined,
      { orderedTools: true },
    ))
      events.push(event)
  } catch (error) {
    thrown = error
  }

  const text = events.flatMap((event) => {
    if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
      return [event.delta.text]
    return []
  })
  const tools = events.filter(
    (event) => event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart',
  )

  expect(thrown).toBe(failure)
  expect(text.join('')).toBe('Keep this text')
  expect(tools).toEqual([])
})

async function collect(chunks: AsyncIterable<StreamChunk>): Promise<ModelStreamEvent[]> {
  const events: ModelStreamEvent[] = []
  for await (const event of readStream(chunks, undefined, undefined, { orderedTools: true }))
    events.push(event)
  return events
}

function streamChunk(fields: Partial<StreamChunk> = {}): StreamChunk {
  return {
    text: fields.text ?? '',
    reasoning: fields.reasoning ?? '',
    calls: fields.calls ?? [],
    stop: fields.stop,
    usage: fields.usage,
  }
}
