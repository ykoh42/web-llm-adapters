import { expect, it, vi } from 'vitest'
import { Message, ReasoningBlock, TextBlock } from '@strands-agents/sdk'
import { trimThoughts } from '../../src/gemma/history.js'
import { chunk, collect, setup } from '../../fixtures/engine.js'

it('keeps message identity and metadata when removing older thoughts from a copy', () => {
  const thought = new ReasoningBlock({ text: 'Earlier thought.' })
  const text = new TextBlock('Earlier answer.')
  const metadata = { custom: { source: 'session-only' } }
  const message = new Message({ role: 'assistant', content: [thought, text], metadata })
  const history = [message, new Message({ role: 'user', content: [new TextBlock('Next')] })]
  const snapshot = history.map((message) => message.toJSON())

  const result = trimThoughts(history)

  expect(result[0]).not.toBe(message)
  expect(result[0]?.trackingId).toBe(message.trackingId)
  expect(result[0]?.metadata).toBe(metadata)
  expect(result[0]?.content).toEqual([text])
  expect(result[1]).toBe(history[1])
  expect(history.map((message) => message.toJSON())).toEqual(snapshot)
})

it.each([false, true])(
  'keeps private message fields out of native input (thinking=%s)',
  async (thinking) => {
    const { model, engine, create } = setup(chunk({ content: 'Answer' }, 'stop'))
    const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield {
          id: 'test',
          object: 'text_completion' as const,
          created: 0,
          model: 'test',
          choices: [{ index: 0, text: 'Answer', finish_reason: 'stop' as const }],
        }
      })(),
    )
    const message = new Message({
      role: 'assistant',
      content: [new ReasoningBlock({ text: 'Earlier thought.' }), new TextBlock('Earlier answer.')],
      metadata: { custom: { secret: 'never-model', score: Infinity } },
    })
    const history = [message, new Message({ role: 'user', content: [new TextBlock('Next')] })]
    const snapshot = history.map((message) => message.toJSON())
    model.updateConfig({
      modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
      request: { extra_body: { enable_thinking: thinking } },
    })

    await collect(model.stream(history))

    if (thinking) {
      expect(create).not.toHaveBeenCalled()
      expect(raw.mock.calls[0]?.[0].prompt).toBe(
        '<|turn>system\n<|think|>\n<turn|>\n<|turn>model\nEarlier answer.<turn|>\n' +
          '<|turn>user\nNext<turn|>\n<|turn>model\n',
      )
    } else {
      expect(raw).not.toHaveBeenCalled()
      expect(create.mock.calls[0]?.[0].messages).toEqual([
        { role: 'assistant', content: 'Earlier answer.' },
        { role: 'user', content: 'Next' },
      ])
    }
    expect(history.map((message) => message.toJSON())).toEqual(snapshot)
  },
)
