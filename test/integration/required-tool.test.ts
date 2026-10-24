import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock, type ModelStreamEvent } from '@strands-agents/sdk'
import { createAgent, toolStrategy } from 'langchain'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  postInitAndCheckFieldsCompletion,
} from '@mlc-ai/web-llm'
import { BrowserModel, WebLLMModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

const models = ['gemma-4-E2B-it-q4f16_1-MLC', 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC']
const clients = ['strands', 'strands-browser', 'langchain', 'langchain-browser']
const schema = {
  type: 'object' as const,
  properties: { code: { type: 'string' as const } },
  required: ['code'],
}
const lookup = { name: 'lookup', description: 'Look up a code.', inputSchema: schema }

afterEach(() => vi.unstubAllGlobals())

for (const client of clients) {
  it.each(models)(
    `${client} maps one required %s tool to the same named request`,
    async (modelId) => {
      const { engine, create } = setupEngine()
      const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async (request) => {
        postInitAndCheckFieldsCompletion(request, modelId)
        return (async function* () {
          yield {
            id: 'raw',
            object: 'text_completion' as const,
            created: 0,
            model: modelId,
            choices: [
              { index: 0, text: 'code:<|"|>A7<|"|>}<tool_call|>', finish_reason: 'stop' as const },
            ],
          }
        })()
      })
      create.mockImplementation(async (request) => {
        postInitAndCheckFieldsChatCompletion(request, modelId, ModelType.LLM)
        return (async function* () {
          yield chunk({ content: '[{"name":"lookup","arguments":{"code":"A7"}}]' })
          yield chunk(
            {
              tool_calls: [
                {
                  index: 0,
                  id: 'native-call',
                  type: 'function',
                  function: { name: 'lookup', arguments: '{"code":"A7"}' },
                },
              ],
            },
            'tool_calls',
          )
        })()
      })
      const first = strandsBridge(engine)
      const second = langchainBridge(engine)
      vi.stubGlobal('window', { webllmBridge: first, webllmLangChainBridge: second })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          return run(structuredClone(arg))
        },
      }
      const config = { modelId, request: { max_tokens: 4096, seed: 7 } }
      const requests: unknown[] = []
      try {
        for (const mode of ['named', 'required']) {
          if (client.startsWith('strands')) {
            const model =
              client === 'strands'
                ? new WebLLMModel({ engine, ...config })
                : new BrowserModel(page, config)
            const output = await collect(
              model.streamAggregated(
                [new Message({ role: 'user', content: [new TextBlock('Look up A7.')] })],
                {
                  toolSpecs: [lookup],
                  toolChoice: mode === 'named' ? { tool: { name: 'lookup' } } : { any: {} },
                },
              ),
            )
            expect(output.filter((part) => part.type === 'toolUseBlock')).toMatchObject([
              { name: 'lookup', input: { code: 'A7' } },
            ])
          } else {
            const model =
              client === 'langchain'
                ? new ChatWebLLM({ engine, ...config })
                : new BrowserChatWebLLM(page, config)
            const output = await model.invoke('Look up A7.', {
              tools: [
                {
                  type: 'function',
                  function: {
                    name: lookup.name,
                    description: lookup.description,
                    parameters: schema,
                  },
                },
              ],
              tool_choice: mode === 'named' ? 'lookup' : 'required',
            })
            expect(output.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
            expect(output.text).toBe('')
          }
          requests.push(
            structuredClone(
              modelId.startsWith('gemma-')
                ? raw.mock.calls.at(-1)![0]
                : create.mock.calls.at(-1)![0],
            ),
          )
        }
        expect(requests[1]).toEqual(requests[0])
        if (modelId.startsWith('gemma-')) {
          expect(raw).toHaveBeenCalledTimes(2)
          expect(create).not.toHaveBeenCalled()
        } else {
          expect(create).toHaveBeenCalledTimes(2)
          expect(raw).not.toHaveBeenCalled()
        }
      } finally {
        await first.close()
        await second.close()
      }
    },
  )
}

for (const client of clients) {
  it.each(models.flatMap((modelId) => [0, 2].map((count) => ({ modelId, count }))))(
    `${client} rejects a required $modelId choice with $count tools before inference`,
    async ({ modelId, count }) => {
      const { engine, create } = setupEngine()
      const raw = vi.spyOn(engine.completions, 'create')
      const first = strandsBridge(engine)
      const second = langchainBridge(engine)
      vi.stubGlobal('window', { webllmBridge: first, webllmLangChainBridge: second })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          return run(structuredClone(arg))
        },
      }
      const tools = count ? [lookup, { ...lookup, name: 'other' }] : []
      try {
        if (client.startsWith('strands')) {
          const model =
            client === 'strands'
              ? new WebLLMModel({ engine, modelId })
              : new BrowserModel(page, { modelId })
          await expect(
            collect(
              model.stream(
                [new Message({ role: 'user', content: [new TextBlock('Look up A7.')] })],
                { toolSpecs: tools, toolChoice: { any: {} } },
              ),
            ),
          ).rejects.toThrow(/tool.?choice/i)
        } else {
          const model =
            client === 'langchain'
              ? new ChatWebLLM({ engine, modelId })
              : new BrowserChatWebLLM(page, { modelId })
          await expect(
            model.invoke('Look up A7.', {
              tools: tools.map((tool) => ({
                type: 'function',
                function: { name: tool.name, description: tool.description, parameters: schema },
              })),
              tool_choice: 'required',
            }),
          ).rejects.toThrow(/tool.?choice/i)
        }
        expect(create).not.toHaveBeenCalled()
        expect(raw).not.toHaveBeenCalled()
      } finally {
        await first.close()
        await second.close()
      }
    },
  )

  it(`${client} rejects an unoffered native tool for a required choice`, async () => {
    const modelId = models[1]!
    const { engine } = setupEngine(
      chunk(
        { tool_calls: [{ index: 0, id: 'wrong', function: { name: 'other', arguments: '{}' } }] },
        'tool_calls',
      ),
    )
    const first = strandsBridge(engine)
    const second = langchainBridge(engine)
    vi.stubGlobal('window', { webllmBridge: first, webllmLangChainBridge: second })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        return run(structuredClone(arg))
      },
    }
    try {
      if (client.startsWith('strands')) {
        const model =
          client === 'strands'
            ? new WebLLMModel({ engine, modelId })
            : new BrowserModel(page, { modelId })
        const events: ModelStreamEvent[] = []
        await expect(
          (async () => {
            for await (const event of model.stream(
              [new Message({ role: 'user', content: [new TextBlock('Look up A7.')] })],
              { toolSpecs: [lookup], toolChoice: { any: {} } },
            ))
              events.push(event)
          })(),
        ).rejects.toThrow('outside tool_choice')
        expect(
          events.some(
            (event) =>
              event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart',
          ),
        ).toBe(false)
      } else {
        const model =
          client === 'langchain'
            ? new ChatWebLLM({ engine, modelId })
            : new BrowserChatWebLLM(page, { modelId })
        await expect(
          model.invoke('Look up A7.', {
            tools: [{ type: 'function', function: { name: 'lookup', parameters: schema } }],
            tool_choice: 'required',
          }),
        ).rejects.toThrow('outside tool_choice')
      }
    } finally {
      await first.close()
      await second.close()
    }
  })
}

for (const modelId of ['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  it.each([1, 2])(`${modelId} keeps its existing required choice with %s tools`, async (count) => {
    const source = modelId.startsWith('Qwen3.5-')
      ? '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
      : '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
    const { engine, create } = setupEngine(chunk({ content: source }, 'stop'))
    const model = new ChatWebLLM({ engine, modelId })
    const output = await model.invoke('Look up A7.', {
      tools: ['lookup', 'other'].slice(0, count).map((name) => ({
        type: 'function',
        function: { name, parameters: schema },
      })),
      tool_choice: 'required',
    })
    expect(output.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
    expect(create.mock.calls[0]![0].messages[0]?.content).toMatch(/must call at least one function/)
    expect(create.mock.calls[0]![0].tool_choice).toBeUndefined()
  })
}

it.each(models)('%s accepts LangChain toolStrategy with one response schema', async (modelId) => {
  const { engine, create } = setupEngine()
  const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async (request) => {
    postInitAndCheckFieldsCompletion(request, modelId)
    return (async function* () {
      yield {
        id: 'raw',
        object: 'text_completion' as const,
        created: 0,
        model: modelId,
        choices: [{ index: 0, text: 'answer:42}<tool_call|>', finish_reason: 'stop' as const }],
      }
    })()
  })
  create.mockImplementation(async (request) => {
    postInitAndCheckFieldsChatCompletion(request, modelId, ModelType.LLM)
    const name = request.tools![0]!.function.name
    return (async function* () {
      yield chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'native-call',
              type: 'function',
              function: { name, arguments: '{"answer":42}' },
            },
          ],
        },
        'tool_calls',
      )
    })()
  })
  const model = new ChatWebLLM({ engine, modelId, request: { max_tokens: 4096, seed: 7 } })
  const agent = createAgent({
    model,
    responseFormat: toolStrategy({
      type: 'object',
      properties: { answer: { type: 'number' } },
      required: ['answer'],
    }),
  })
  const result = await agent.invoke({ messages: [{ role: 'user', content: 'Return answer 42.' }] })
  expect(result.structuredResponse).toEqual({ answer: 42 })
  expect(raw.mock.calls.length + create.mock.calls.length).toBe(1)
})
