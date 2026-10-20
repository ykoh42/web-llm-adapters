import { expect, it, vi } from 'vitest'
import type { Completion } from '@mlc-ai/web-llm'
import { Message, TextBlock } from '@strands-agents/sdk'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, collect, pause, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const tools = [{ name: 'lookup', description: 'Look up a code.' }]
const definition = {
  type: 'function',
  function: { name: 'lookup', description: 'Look up a code.' },
}

const messages = [new Message({ role: 'user', content: [new TextBlock('Find A7.')] })]
const cases = [
  {
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    thought: '<think>Plan.</think>',
    call: '<tool_call>{"name":"lookup","arguments":{"code":"A7',
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    thought: '<think>Plan.</think>',
    call: '<tool_call><function=lookup><parameter=code>\nA7',
  },
  {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    thought: '<|channel>thought\nPlan.<channel|>',
    call: '<|tool_call>call:lookup{code:<|"|>A7',
  },
]

function raw(text: string, finish: 'stop' | null = null): Completion {
  return {
    id: 'test',
    model: 'gemma-4-E2B-it-q4f16_1-MLC',
    object: 'text_completion',
    created: 0,
    choices: [{ index: 0, text, finish_reason: finish }],
  }
}

async function* source<T>(values: T[], failure?: Error) {
  yield* values
  if (failure) throw failure
}

for (const client of ['strands', 'langchain']) {
  for (const finish of ['error', 'end', 'invalid']) {
    it.each(cases)(
      `${client} keeps $modelId output before a stream ${finish}`,
      async ({ modelId, thought, call }) => {
        const toolText = finish === 'invalid' ? call + '</tool' : call
        const text = 'Before<' + thought + toolText
        for (const size of [1, 3, text.length]) {
          const pieces = text.match(new RegExp(`[\\s\\S]{1,${size}}`, 'g'))!
          const failure = finish === 'error' ? new Error('Native stream failed') : undefined
          const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
          if (modelId.startsWith('gemma-')) {
            vi.spyOn(engine.completions, 'create')
              .mockImplementation(async () => source([raw('Recovered', 'stop')]))
              .mockResolvedValueOnce(
                source(
                  pieces.map((text, index) =>
                    raw(text, finish === 'invalid' && index === pieces.length - 1 ? 'stop' : null),
                  ),
                  failure,
                ),
              )
          } else
            create.mockResolvedValueOnce(
              source(
                pieces.map((content, index) =>
                  chunk(
                    { content },
                    finish === 'invalid' && index === pieces.length - 1 ? 'stop' : null,
                  ),
                ),
                failure,
              ),
            )
          const config = { modelId, request: { extra_body: { enable_thinking: true } } }
          const strands = new WebLLMModel({ engine, ...config })
          const langchain = new ChatWebLLM({ engine, ...config })
          const received: { type: string; value: string }[] = []
          function append(type: string, value: string) {
            if (!value) return
            const last = received.at(-1)
            if (last?.type === type) last.value += value
            else received.push({ type, value })
          }
          const pending = (async () => {
            if (client === 'strands') {
              for await (const event of strands.stream(messages, { toolSpecs: tools })) {
                if (event.type === 'modelContentBlockStartEvent')
                  expect(event.start).toBeUndefined()
                if (event.type === 'modelContentBlockDeltaEvent') {
                  if (event.delta.type === 'textDelta') append('text', event.delta.text)
                  if (event.delta.type === 'reasoningContentDelta')
                    append('reasoning', event.delta.text ?? '')
                }
                expect(event.type).not.toBe('modelMessageStopEvent')
              }
            } else {
              for await (const item of await langchain.bindTools([definition]).stream('Find A7.')) {
                expect(item.tool_calls ?? []).toEqual([])
                for (const block of item.contentBlocks) {
                  if (block.type === 'text') append('text', block.text)
                  if (block.type === 'reasoning') append('reasoning', block.reasoning)
                }
              }
            }
          })()
          if (failure) await expect(pending).rejects.toBe(failure)
          else if (finish === 'invalid') await expect(pending).rejects.toThrow('tool')
          else await expect(pending).rejects.toThrow('without a stop reason')
          expect(received).toEqual([
            { type: 'text', value: 'Before<' },
            { type: 'reasoning', value: 'Plan.' },
            { type: 'text', value: toolText },
          ])
          if (client === 'strands') {
            expect(await collect(strands.stream(messages))).toContainEqual({
              type: 'modelContentBlockDeltaEvent',
              delta: { type: 'textDelta', text: 'Recovered' },
            })
          } else expect((await langchain.invoke('Try again.')).text).toBe('Recovered')
        }
      },
    )
  }
}

for (const client of ['strands', 'langchain']) {
  for (const phase of ['waiting', 'callback', 'buffered']) {
    it.each([
      ...cases,
      {
        modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
        thought: 'Before',
        call: '[{"name":"lookup","arguments":{',
      },
    ])(
      `${client} cancels $modelId during ${phase} without observing late data`,
      async ({ modelId, thought, call }) => {
        const { engine, create, stop } = setup(chunk({ content: 'Recovered' }, 'stop'))
        const waiting = pause()
        const hold = pause()
        let closed = false
        async function* delayed<T>(first: T, late: T) {
          try {
            yield first
            if (phase === 'buffered') throw new Error('Native stream failed')
            waiting.resume()
            await hold.wait
            yield late
          } finally {
            closed = true
          }
        }
        const gemma = modelId.startsWith('gemma-')
        if (gemma) {
          const first = raw(phase === 'buffered' ? call : thought)
          if (phase === 'callback') first.choices[0]!.index = 1
          vi.spyOn(engine.completions, 'create')
            .mockImplementation(async () => source([raw('Recovered', 'stop')]))
            .mockResolvedValueOnce(delayed(first, raw(call, 'stop')))
        } else {
          const first = chunk({ content: phase === 'buffered' ? call : thought })
          if (phase === 'callback') first.choices[0]!.index = 1
          create.mockResolvedValueOnce(delayed(first, chunk({ content: call }, 'stop')))
        }
        stop.mockImplementation(async () => hold.resume())
        const controller = new AbortController()
        const onChunk = vi.fn()
        if (phase === 'callback') onChunk.mockImplementationOnce(() => controller.abort())
        const config = {
          modelId,
          onChunk,
          ...(gemma && { request: { extra_body: { enable_thinking: true } } }),
        }
        const strands = new WebLLMModel({ engine, ...config })
        const langchain = new ChatWebLLM({ engine, ...config })
        const pending =
          client === 'strands'
            ? (async () => {
                const events = []
                for await (const event of strands.stream(messages, {
                  toolSpecs: tools,
                  cancelSignal: controller.signal,
                })) {
                  events.push(event)
                  if (phase === 'buffered' && event.type === 'modelContentBlockDeltaEvent')
                    controller.abort()
                }
                return events
              })()
            : (async () => {
                const events = []
                for await (const event of await langchain
                  .bindTools([definition])
                  .stream('Find A7.', { signal: controller.signal })) {
                  events.push(event)
                  if (phase === 'buffered') controller.abort()
                }
                return events
              })()
        const cancelled =
          client === 'strands'
            ? expect(pending).resolves.toContainEqual({
                type: 'modelMessageStopEvent',
                stopReason: 'cancelled',
              })
            : expect(pending).rejects.toHaveProperty('name', 'AbortError')
        if (phase === 'waiting') {
          await waiting.wait
          controller.abort()
        }
        await cancelled
        expect(onChunk).toHaveBeenCalledTimes(1)
        expect(closed).toBe(true)
        if (client === 'strands') {
          expect(await collect(strands.stream(messages))).toContainEqual({
            type: 'modelContentBlockDeltaEvent',
            delta: { type: 'textDelta', text: 'Recovered' },
          })
        } else expect((await langchain.invoke('Try again.')).text).toBe('Recovered')
      },
    )
  }
}
