import { Agent, FunctionTool } from '@strands-agents/sdk'
import { expect, it, vi } from 'vitest'
import { withEngine } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from './helpers.js'

it('keeps the original engine and serializes native work with Strands in both directions', async () => {
  const { model, engine, create } = setup()
  const entered = pause()
  const finish = pause()
  const order: string[] = []
  const native = withEngine(engine, async (current) => {
    expect(current).toBe(engine)
    order.push('native')
    entered.resume()
    await finish.wait
    return 42
  })
  await entered.wait
  create.mockImplementation(async () => {
    order.push('agent')
    return chunks(chunk({ content: 'Hello' }, 'stop'))
  })
  const agent = collect(model.stream([]))
  expect(create).not.toHaveBeenCalled()
  const later = withEngine(engine, () => order.push('later'))
  finish.resume()
  expect(await native).toBe(42)
  await agent
  await later
  expect(order).toEqual(['native', 'agent', 'later'])

  const stream = model.stream([])
  await stream.next()
  const waiting = vi.fn()
  const queued = withEngine(engine, waiting)
  await Promise.resolve()
  expect(waiting).not.toHaveBeenCalled()
  await stream.return(undefined)
  await queued
  expect(waiting).toHaveBeenCalledOnce()
})

it('lets a Strands tool use native embeddings on the same engine', async () => {
  const { engine, model, create } = setup()
  create.mockResolvedValueOnce(
    chunks(
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'embed-1',
              type: 'function',
              function: { name: 'embed', arguments: '{}' },
            },
          ],
        },
        'tool_calls',
      ),
    ),
  )
  create.mockResolvedValueOnce(chunks(chunk({ content: 'Done' }, 'stop')))
  const result = {
    object: 'list' as const,
    model: 'embedding-model',
    data: [{ object: 'embedding' as const, index: 0, embedding: [0.2, 0.4] }],
    usage: { prompt_tokens: 2, total_tokens: 2, extra: { prefill_tokens_per_s: 10 } },
  }
  const embed = vi.spyOn(engine.embeddings, 'create').mockResolvedValue(result)
  const tool = new FunctionTool({
    name: 'embed',
    description: 'Embed text',
    callback: () =>
      withEngine(engine, async (engine) => {
        const result = await engine.embeddings.create({
          model: 'embedding-model',
          input: 'hello',
        })
        return result.data[0]!.embedding
      }),
  })
  const answer = await new Agent({ model, tools: [tool], printer: false }).invoke('Embed hello')
  expect(String(answer)).toBe('Done')
  expect(embed).toHaveBeenCalledExactlyOnceWith({ model: 'embedding-model', input: 'hello' })
  expect(JSON.stringify(create.mock.calls[1]![0].messages)).toContain('0.2')
})

it('keeps a native stream in its turn until the callback finishes consuming it', async () => {
  const { engine, create } = setup()
  const entered = pause()
  const finish = pause()
  create.mockResolvedValueOnce(
    (async function* () {
      entered.resume()
      yield chunk({ content: 'Hi' })
      await finish.wait
      yield chunk({}, 'stop')
    })(),
  )
  const running = withEngine(engine, async (engine) =>
    collect(
      await engine.chat.completions.create({
        messages: [],
        stream: true,
      }),
    ),
  )
  await entered.wait
  const next = vi.fn()
  const queued = withEngine(engine, next)
  await Promise.resolve()
  expect(next).not.toHaveBeenCalled()
  finish.resume()
  expect(await running).toHaveLength(2)
  await queued
  expect(next).toHaveBeenCalledOnce()
})

it('closes an escaped iterator before releasing the engine', async () => {
  const { engine, stop } = setup()
  const returned = vi.fn(async () => ({ done: true as const, value: undefined }))
  const source = {
    [Symbol.asyncIterator]: () => ({
      next: async () => ({ done: false, value: 1 }),
      return: returned,
    }),
  }
  await expect(withEngine(engine, () => source)).rejects.toThrow('Consume native streams inside')
  expect(returned).toHaveBeenCalledOnce()
  expect(stop).toHaveBeenCalledOnce()
  expect(await withEngine(engine, () => 'next')).toBe('next')
})

it('preserves native errors and releases the engine after a failed callback', async () => {
  const { engine } = setup()
  const failure = new Error('Native failure')
  await expect(
    withEngine(engine, () => {
      throw failure
    }),
  ).rejects.toBe(failure)
  expect(await withEngine(engine, () => 'next')).toBe('next')
})
