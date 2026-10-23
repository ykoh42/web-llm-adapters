import { afterEach, describe, expect, it, vi } from 'vitest'
import { InMemoryCache } from '@langchain/core/caches'
import { tool } from '@langchain/core/tools'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import { z } from 'zod'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, chunks, setup } from '../fixtures/engine.js'

const bridges: BrowserBridge[] = []
afterEach(async () => {
  try {
    await Promise.all(bridges.splice(0).map((bridge) => bridge.close()))
  } finally {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  }
})

describe.each(['local', 'browser'])('%s LangChain cache', (transport) => {
  function cachedModel() {
    const native = setup()
    const bridge = createBrowserBridge(native.engine)
    bridges.push(bridge)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        const result = await run(structuredClone(arg))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const options = { modelId: 'test-model', cache: new InMemoryCache() }
    return {
      ...native,
      page,
      model:
        transport === 'browser'
          ? new BrowserChatWebLLM(page, options)
          : new ChatWebLLM({ engine: native.engine, ...options }),
    }
  }

  it('keeps cached interpretations separate for a prefilled thinking template', async () => {
    const native = cachedModel()
    const cache = new InMemoryCache()
    const source = 'Plan.\n</think>42'
    native.create.mockImplementation(async () => chunks(chunk({ content: source }, 'stop')))
    function model(thinkingPrefilled?: boolean) {
      const options = {
        modelId: 'Qwen3.5-4B-q4f16_1-MLC',
        cache,
        ...(thinkingPrefilled !== undefined && { thinkingPrefilled }),
      }
      return transport === 'browser'
        ? new BrowserChatWebLLM(native.page, options)
        : new ChatWebLLM({ engine: native.engine, ...options })
    }
    const plain = model()
    const prefilled = model(true)
    const disabled = model(false)
    for (let repeat = 0; repeat < 2; repeat++) {
      expect((await plain.invoke('Same input.')).text).toBe(source)
      expect((await prefilled.invoke('Same input.')).text).toBe('42')
      expect((await disabled.invoke('Same input.')).text).toBe(source)
    }
    expect(native.create).toHaveBeenCalledTimes(2)
    expect(native.create.mock.calls[1]?.[0]).toEqual(native.create.mock.calls[0]?.[0])
  })

  it.each(['name', 'description', 'schema', 'order'])(
    'separates cached answers when tool %s changes',
    async (change) => {
      const { model, create } = cachedModel()
      create.mockResolvedValueOnce(chunks(chunk({ content: 'First answer' }, 'stop')))
      create.mockResolvedValueOnce(chunks(chunk({ content: 'Second answer' }, 'stop')))
      const first = tool(() => '', {
        name: 'lookup',
        description: 'Look up a code.',
        schema: z.object({ code: z.string() }),
      })
      const second = tool(() => '', {
        name: change === 'name' ? 'search' : 'lookup',
        description: change === 'description' ? 'Read an access code.' : 'Look up a code.',
        schema:
          change === 'schema' ? z.object({ code: z.number() }) : z.object({ code: z.string() }),
      })
      const other = tool(() => '', { name: 'other', schema: z.object({}) })
      const firstTools = change === 'order' ? [first, other] : [first]
      const secondTools = change === 'order' ? [other, second] : [second]
      const before = JSON.stringify([convertToOpenAITool(first), convertToOpenAITool(second)])
      const a = model.bindTools(firstTools)
      const b = model.bindTools(secondTools)
      for (let repeat = 0; repeat < 2; repeat++) {
        expect((await a.invoke('Choose a tool.')).text).toBe('First answer')
        expect((await b.invoke('Choose a tool.')).text).toBe('Second answer')
      }
      expect(create).toHaveBeenCalledTimes(2)
      expect(create.mock.calls[0]?.[0].tools).not.toEqual(create.mock.calls[1]?.[0].tools)
      expect(JSON.stringify([convertToOpenAITool(first), convertToOpenAITool(second)])).toBe(before)
    },
  )

  it('reuses a cached answer for equivalent framework and native tools', async () => {
    const { model, create } = cachedModel()
    create.mockImplementation(async () => chunks(chunk({ content: 'Answer' }, 'stop')))
    const lookup = tool(() => '', { name: 'lookup', schema: z.object({ code: z.string() }) })
    for (const definition of [lookup, convertToOpenAITool(lookup), lookup])
      expect((await model.bindTools([definition]).invoke('Choose a tool.')).text).toBe('Answer')
    expect(create).toHaveBeenCalledOnce()
  })

  it('rejects duplicate tool names even when a valid tool list already populated the cache', async () => {
    const { model, create } = cachedModel()
    create.mockImplementation(async () => chunks(chunk({ content: 'Answer' }, 'stop')))
    const lookup = tool(() => '', { name: 'lookup', schema: z.object({}) })
    const search = tool(() => '', { name: 'search', schema: z.object({}) })
    expect((await model.invoke('Choose a tool.', { tools: [lookup, search] })).text).toBe('Answer')
    await expect(model.invoke('Choose a tool.', { tools: [lookup, lookup] })).rejects.toThrow(
      'unique',
    )
    expect(create).toHaveBeenCalledOnce()
  })

  it.each([
    { first: 'v0', source: 'constructor' },
    { first: 'v1', source: 'constructor' },
    { first: 'v0', source: 'environment' },
    { first: 'v1', source: 'environment' },
  ] as const)(
    'keeps output formats separate when $first from $source populates the cache first',
    async ({ first, source }) => {
      const { engine, create } = setup(
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: 'lookup-call',
                function: { name: 'lookup', arguments: '{"code":"A7"}' },
              },
            ],
          },
          'tool_calls',
        ),
      )
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const value = await run(structuredClone(arg))
          return value === undefined ? value : JSON.parse(JSON.stringify(value))
        },
      }
      if (transport === 'browser') {
        const bridge = createBrowserBridge(engine)
        bridges.push(bridge)
        vi.stubGlobal('window', { webllmLangChainBridge: bridge })
      }
      function model(outputVersion: 'v0' | 'v1', cache: InMemoryCache | false) {
        if (source === 'environment') vi.stubEnv('LC_OUTPUT_VERSION', outputVersion)
        const options = {
          modelId: 'test-model',
          cache,
          ...(source === 'constructor' && { outputVersion }),
        }
        const chat =
          transport === 'browser'
            ? new BrowserChatWebLLM(page, options)
            : new ChatWebLLM({ engine, ...options })
        return chat.bindTools([
          {
            type: 'function',
            function: {
              name: 'lookup',
              parameters: { type: 'object', properties: { code: { type: 'string' } } },
            },
          },
        ])
      }
      const input = 'Look up A7.'
      const versions = [first, first === 'v0' ? 'v1' : 'v0'] as const
      const expected = await Promise.all(
        versions.map((version) => model(version, false).invoke(input)),
      )
      expect(expected[0]!.content).not.toEqual(expected[1]!.content)
      const cache = new InMemoryCache()
      for (let repeat = 0; repeat < 2; repeat++) {
        for (const [index, version] of versions.entries()) {
          const answer = await model(version, cache).invoke(input)
          expect(answer.content).toEqual(expected[index]!.content)
          expect(answer.tool_calls).toEqual(expected[index]!.tool_calls)
        }
      }
      for (const [index, version] of versions.entries()) {
        const other = version === 'v0' ? 'v1' : 'v0'
        const answer = await model(other, cache).invoke(input, { outputVersion: version })
        expect(answer.content).toEqual(expected[index]!.content)
        expect(answer.tool_calls).toEqual(expected[index]!.tool_calls)
        for (const outputVersion of [undefined, null]) {
          const options = {}
          Object.assign(options, { outputVersion })
          const inherited = await model(version, cache).invoke(input, options)
          expect(inherited.content).toEqual(expected[index]!.content)
          expect(inherited.tool_calls).toEqual(expected[index]!.tool_calls)
        }
      }
      expect(create).toHaveBeenCalledTimes(4)
      for (const [request] of create.mock.calls) expect(request).toEqual(create.mock.calls[0]![0])
    },
  )
})
