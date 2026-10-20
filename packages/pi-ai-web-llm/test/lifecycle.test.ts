import { describe, expect, it, vi } from 'vitest'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { Agent } from '@earendil-works/pi-agent-core'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import { chunk, chunks, setup } from './helpers.js'

const context = { messages: [{ role: 'user' as const, content: 'Hello', timestamp: 0 }] }

describe('Pi provider lifecycle', () => {
  it.each([false, true])(
    'closes a native stream aborted during creation (observer=%s)',
    async (observe) => {
      const { provider, model, create } = setup()
      const next = vi.fn(async () => ({
        done: false as const,
        value: chunk({ content: 'Late' }, 'stop'),
      }))
      const close = vi.fn(async () => ({ done: true as const, value: undefined }))
      const source: AsyncIterable<ChatCompletionChunk> = {
        [Symbol.asyncIterator]: () => ({ next, return: close }),
      }
      let resolve!: (value: AsyncIterable<ChatCompletionChunk>) => void
      create.mockReturnValueOnce(
        new Promise((done) => {
          resolve = done
        }),
      )
      const controller = new AbortController()
      const stream = provider.streamSimple(model, normalizeContext(context), {
        signal: controller.signal,
        ...(observe && { onProviderStreamEvent: async () => {} }),
      })
      await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(1))
      controller.abort()
      resolve(source)
      expect((await stream.result()).stopReason).toBe('aborted')
      expect(close).toHaveBeenCalledTimes(1)
      expect(next).not.toHaveBeenCalled()
    },
  )

  it('maps Pi Agent thinking high then off to native true then false', async () => {
    const { models, model, create } = setup(undefined, { modelId: 'Qwen3-8B-q4f16_1-MLC' })
    const agent = new Agent({
      initialState: { model, thinkingLevel: 'high' },
      streamFn: models.streamSimple.bind(models),
    })
    await agent.prompt('Hello')
    agent.state.thinkingLevel = 'off'
    await agent.prompt('Hello again')
    expect(create.mock.calls.map(([request]) => request.extra_body?.enable_thinking)).toEqual([
      true,
      false,
    ])
  })

  it('preserves native thinking defaults for full stream calls', async () => {
    const { models, model, create } = setup(undefined, {
      modelId: 'Qwen3-8B-q4f16_1-MLC',
      request: { extra_body: { enable_thinking: true } },
    })
    await models.complete(model, context)
    await models.completeSimple(model, context)
    expect(create.mock.calls.map(([request]) => request.extra_body?.enable_thinking)).toEqual([
      true,
      false,
    ])
  })

  it('awaits payload transformations and native chunk observers in order', async () => {
    const { models, model, create } = setup()
    const order: string[] = []
    const result = await models.completeSimple(model, context, {
      onPayload: async (payload) => {
        await Promise.resolve()
        order.push('payload')
        return { ...(payload as object), temperature: 0.25 }
      },
      onProviderStreamEvent: async () => {
        await Promise.resolve()
        order.push('chunk')
      },
    })
    expect(result.stopReason).toBe('stop')
    expect(create.mock.calls[0]?.[0]).toHaveProperty('temperature', 0.25)
    expect(order).toEqual(['payload', 'chunk'])
  })

  it('does not open the engine after a payload callback aborts', async () => {
    const { models, model, create } = setup()
    const controller = new AbortController()
    const result = await models.completeSimple(model, context, {
      signal: controller.signal,
      onPayload: () => {
        controller.abort()
      },
    })
    expect(result.stopReason).toBe('aborted')
    expect(create).not.toHaveBeenCalled()
  })

  it('rejects a payload replacement that changes the loaded model', async () => {
    const { models, model, create } = setup()
    const result = await models.completeSimple(model, context, {
      onPayload: () => ({ model: 'other', stream: true, n: 1 }),
    })
    expect(result.stopReason).toBe('error')
    expect(create).not.toHaveBeenCalled()
  })

  it('closes the native iterator after an awaited observer fails', async () => {
    const { models, model, create } = setup()
    const close = vi.fn(async () => ({ done: true as const, value: undefined }))
    create.mockResolvedValueOnce({
      [Symbol.asyncIterator]: () => ({
        next: async () => ({ done: false as const, value: chunk({ content: 'Hi' }) }),
        return: close,
      }),
    })
    const result = await models.completeSimple(model, context, {
      onProviderStreamEvent: async () => {
        throw new Error('Observer failed')
      },
    })
    expect(result).toMatchObject({ stopReason: 'error', errorMessage: 'Observer failed' })
    expect(close).toHaveBeenCalledTimes(1)
    expect((await models.completeSimple(model, context)).stopReason).toBe('stop')
  })

  it('keeps samplingParams precedence without allowing it to replace the conversation', async () => {
    const { models, model, create } = setup()
    await models.completeSimple(model, context, {
      temperature: 0.4,
      maxTokens: 50,
      samplingParams: { temperature: 0.7, max_tokens: 80 },
    })
    expect(create.mock.calls[0]?.[0]).toMatchObject({ temperature: 0.7, max_tokens: 80 })
    expect(
      (await models.completeSimple(model, context, { samplingParams: { messages: [] } }))
        .stopReason,
    ).toBe('error')
  })

  it('rejects output chunks following the native stop reason', async () => {
    const { models, model, create } = setup()
    create.mockResolvedValueOnce(
      chunks(chunk({ content: 'Hello' }, 'stop'), chunk({ content: 'Late' })),
    )
    expect((await models.completeSimple(model, context)).stopReason).toBe('error')
  })
})
