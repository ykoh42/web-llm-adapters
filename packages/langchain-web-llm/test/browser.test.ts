import { afterEach, describe, expect, it, vi } from 'vitest'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { BrowserChatWebLLM, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

function browser(...items: ReturnType<typeof chunk>[]) {
  const result = setup(...items)
  const bridge = createBrowserBridge(result.engine)
  vi.stubGlobal('window', { webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  return {
    ...result,
    bridge,
    page,
    model: new BrowserChatWebLLM(page, { modelId: 'Qwen3-8B-q4f16_1-MLC' }),
  }
}
afterEach(() => vi.unstubAllGlobals())

describe('browser transport', () => {
  it('runs a Node agent and preserves reasoning, tool IDs and tool results over JSON', async () => {
    const { model, create } = browser()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Use lookup.</think><tool_call>\n{"name":"lookup","arguments":{"code":"A7"}}\n</tool_call>',
          },
          'stop',
        ),
      ),
    )
    create.mockResolvedValueOnce(chunks(chunk({ content: '<think>Done.</think>42' }, 'stop')))
    const lookup = tool(({ code }) => `Answer ${code}: 42`, {
      name: 'lookup',
      description: 'Get an answer.',
      schema: z.object({ code: z.string() }),
    })
    const agent = createAgent({ model, tools: [lookup] })
    const result = await agent.invoke({ messages: [{ role: 'user', content: 'Look up A7.' }] })
    expect(result.messages.at(-1)?.text).toBe('42')
    expect(result.messages[1]?.content).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'Use lookup.' }),
    )
    expect(JSON.stringify(create.mock.calls[1]?.[0])).toContain('Answer A7: 42')
  })

  it('returns native chunks and usage metadata without sending callbacks over JSON', async () => {
    const usage = {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 2,
        decode_tokens_per_s: 3,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.2,
      },
    }
    const native = { ...chunk({ content: 'Okay' }, 'stop'), usage }
    const { page } = browser(native)
    const onChunk = vi.fn()
    const model = new BrowserChatWebLLM(page, { modelId: 'test-model', onChunk })
    const result = await model.invoke('Hi')
    expect(result.usage_metadata).toMatchObject({
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
    })
    expect(onChunk).toHaveBeenCalledExactlyOnceWith(native)
  })

  it('passes browser errors through and releases the failed stream', async () => {
    const { model, create, bridge } = browser(chunk({ content: 'Recovered' }, 'stop'))
    create.mockRejectedValueOnce(
      Object.assign(new Error('Context full'), { name: 'ContextWindowSizeExceededError' }),
    )
    await expect(model.invoke('Hi')).rejects.toMatchObject({
      name: 'ContextWindowSizeExceededError',
      message: 'Context full',
    })
    expect((await model.invoke('After')).text).toBe('Recovered')
    await bridge.close()
    await expect(model.invoke('Closed')).rejects.toThrow('closed')
    expect(await bridge.next('unknown')).toMatchObject({
      error: { message: 'Unknown browser call.' },
    })
  })

  it('closes without unloading the caller engine', async () => {
    const { bridge, engine } = browser()
    const unload = vi.spyOn(engine, 'unload')
    await bridge.close()
    await bridge.close()
    expect(unload).not.toHaveBeenCalled()
    await expect(
      bridge.start({ config: { modelId: 'test' }, messages: [], options: {} }),
    ).rejects.toThrow('closed')
  })

  it('cancels browser generation and recovers on the next request', async () => {
    const { model, create, stop } = browser(chunk({ content: 'Recovered' }, 'stop'))
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'Started' })
        await hold.wait
        yield chunk({}, 'abort')
      })(),
    )
    stop.mockImplementation(async () => hold.resume())
    const cancel = new AbortController()
    const stream = await model.stream('Hi', { signal: cancel.signal })
    expect((await stream.next()).value?.text).toBe('Started')
    cancel.abort()
    await expect(collect(stream)).rejects.toBeDefined()
    expect((await model.invoke('After')).text).toBe('Recovered')
  })
})

it('uses browser embeddings in LangGraph memory and waits for native cleanup', async () => {
  const { BrowserWebLLMEmbeddings } = await import('../src/index.js')
  const { InMemoryStore } = await import('@langchain/langgraph')
  const { page, engine, bridge } = browser()
  const create = vi.spyOn(engine.embeddings, 'create').mockImplementation(async (request) => ({
    object: 'list',
    model: 'embedding',
    usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
    data: (request.input as string[]).map((text, index) => ({
      object: 'embedding',
      index,
      embedding: text.includes('orchid') ? [1, 0] : [0, 1],
    })),
  }))
  const embeddings = new BrowserWebLLMEmbeddings(page, {
    modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
  })
  expect(await embeddings.embedQuery('orchid')).toEqual([1, 0])
  expect(create.mock.calls[0]?.[0].input).toEqual([
    'Represent this sentence for searching relevant passages: orchid',
  ])
  const store = new InMemoryStore({ index: { embeddings, dims: 2 } })
  await store.put(['notes'], 'flower', { text: 'An orchid.' })
  await store.put(['notes'], 'car', { text: 'A car.' })
  expect((await store.search(['notes'], { query: 'orchid', limit: 1 }))[0]?.key).toBe('flower')
  const hold = pause()
  create.mockImplementationOnce(async () => {
    await hold.wait
    return {
      object: 'list',
      model: 'embedding',
      usage: { prompt_tokens: 1, total_tokens: 1, extra: { prefill_tokens_per_s: 1 } },
      data: [{ object: 'embedding', index: 0, embedding: [1, 0] }],
    }
  })
  const pending = embeddings.embedQuery('orchid')
  await new Promise((resolve) => setTimeout(resolve, 0))
  let closed = false
  const closing = bridge.close().then(() => {
    closed = true
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(closed).toBe(false)
  hold.resume()
  await pending
  await closing
  await expect(embeddings.embedQuery('orchid')).rejects.toThrow('closed')
})
