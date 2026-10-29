import { describe, expect, it, vi } from 'vitest'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'
import { chunk, chunks, collect } from '../../../../fixtures/web-llm.js'

const prefix = '<think>\n\n</think>\n\n'
const options = { parseProtocol: false, thinkingDisabled: true }

it('removes only the native prefix for a non-thinking Qwen model', async () => {
  const options = { thinkingDisabled: true, parseThinking: false }
  const body = '<think> is literal.'
  const result = await collect(
    qwenChunks(chunks(chunk({ content: prefix + body }, 'stop')), [], options),
  )
  expect(result.map((part) => part.text).join('')).toBe(body)
  expect(result.map((part) => part.reasoning).join('')).toBe('')

  const partial = prefix.slice(0, -1)
  const stopped = await collect(
    qwenChunks(chunks(chunk({ content: partial }, 'length')), [], options),
  )
  expect(stopped.map((part) => part.text).join('')).toBe(partial)
})

for (const [name, parse] of [
  ['JSON', qwenChunks],
  ['XML', qwen35Chunks],
] as const) {
  describe(`Qwen ${name} structured prefix`, () => {
    it('keeps literal thinking tags after the disabled prefix across every split', async () => {
      const body = 'Example: <think>literal</think>.'
      const source = prefix + body
      for (let at = 0; at <= source.length; at++) {
        const input = [
          chunk({ content: source.slice(0, at) }),
          chunk({ content: source.slice(at) }, 'stop'),
        ]
        const onChunk = vi.fn()
        const result = await collect(
          parse(chunks(...input), [], { thinkingDisabled: true, onChunk }),
        )
        expect(result.map((part) => part.text).join('')).toBe('\n\n' + body)
        expect(result.map((part) => part.reasoning).join('')).toBe('\n\n')
        expect(onChunk.mock.calls.map(([item]) => item)).toEqual(input)
      }
    })

    it('keeps an unfinished disabled prefix when generation ends', async () => {
      const text = prefix.slice(0, -1)
      const result = await collect(
        parse(chunks(chunk({ content: text }, 'length')), [], { thinkingDisabled: true }),
      )
      expect(result.map((part) => part.text).join('')).toBe(text)
      expect(result.map((part) => part.reasoning).join('')).toBe('')
      expect(result.at(-1)?.stop).toBe('maxTokens')
    })

    it.each([
      ['stop', 'endTurn'],
      ['length', 'maxTokens'],
      ['abort', 'cancelled'],
      [null, undefined],
    ] as const)('preserves %s and usage across every prefix split', async (finish, stop) => {
      const body = ' {"value":"<think>literal</think>"} '
      const usage = {
        prompt_tokens: 10,
        completion_tokens: 7,
        total_tokens: 17,
        extra: {
          e2e_latency_s: 1,
          prefill_tokens_per_s: 1,
          decode_tokens_per_s: 1,
          time_to_first_token_s: 1,
          time_per_output_token_s: 1,
        },
      }
      for (let at = 0; at <= prefix.length; at++) {
        const meta = { ...chunk(), choices: [], usage }
        const input = [
          chunk({ content: prefix.slice(0, at) }),
          meta,
          chunk({ content: prefix.slice(at) + body }, finish),
        ]
        const onChunk = vi.fn()
        const result = await collect(parse(chunks(...input), [], { ...options, onChunk }))
        expect(result.map((part) => part.text).join('')).toBe(body)
        expect(result.map((part) => part.reasoning).join('')).toBe('')
        expect(result.flatMap((part) => part.calls)).toEqual([])
        expect(result.filter((part) => part.usage).map((part) => part.usage)).toEqual([usage])
        expect(result.at(-1)?.stop).toBe(stop)
        expect(onChunk.mock.calls.map(([item]) => item)).toEqual(input)
      }
    })

    it.each(['stop', 'length', 'abort', null] as const)(
      'preserves a partial prefix on %s',
      async (finish) => {
        const text = prefix.slice(0, -1)
        const result = await collect(parse(chunks(chunk({ content: text }, finish)), [], options))
        expect(result.map((part) => part.text).join('')).toBe(text)
      },
    )

    it.each([prefix.slice(0, -1), prefix + '{"partial":'])(
      'preserves received content and the original error for %j',
      async (text) => {
        const failure = new Error('Native stream failed')
        async function* source() {
          yield chunk({ content: text })
          throw failure
        }
        const received: string[] = []
        const pending = (async () => {
          for await (const part of parse(source(), [], options)) received.push(part.text)
        })()
        await expect(pending).rejects.toBe(failure)
        expect(received.join('')).toBe(text.startsWith(prefix) ? text.slice(prefix.length) : text)
      },
    )

    it('observes the original chunk before callback cancellation', async () => {
      const controller = new AbortController()
      const input = chunk({ content: prefix + '{"answer":42}' }, 'stop')
      const onChunk = vi.fn(() => controller.abort())
      const result = await collect(
        parse(chunks(input), [], { ...options, onChunk, signal: controller.signal }),
      )
      expect(result).toEqual([])
      expect(onChunk).toHaveBeenCalledExactlyOnceWith(input)
    })
  })
}
