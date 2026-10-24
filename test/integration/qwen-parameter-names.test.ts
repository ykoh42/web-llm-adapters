import { afterEach, expect, it, vi } from 'vitest'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import {
  Message,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
  type JSONValue,
} from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { WebLLMModel, BrowserModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  ChatWebLLM,
  BrowserChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

afterEach(() => vi.unstubAllGlobals())

const clients = ['strands', 'strands-browser', 'langchain', 'langchain-browser', 'pi'] as const
const modelId = 'Qwen3.5-4B-q4f16_1-MLC'

function client(kind: (typeof clients)[number]) {
  const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
  const strands = strandsBridge(engine)
  const langchain = langchainBridge(engine)
  vi.stubGlobal('window', { webllmBridge: strands, webllmLangChainBridge: langchain })
  const page: BrowserPage = {
    async evaluate(run, input) {
      const result = await run(structuredClone(input))
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  async function invoke(input?: Record<string, JSONValue>) {
    if (kind === 'pi') {
      const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
      const model = provider.getModels()[0]!
      const response = await provider
        .stream(
          model,
          normalizeContext({
            messages: [
              { role: 'user', content: 'Read a value.', timestamp: 0 },
              ...(input
                ? [
                    {
                      role: 'assistant' as const,
                      api: 'webllm' as const,
                      provider: 'webllm',
                      model: modelId,
                      content: [
                        { type: 'toolCall' as const, id: 'call', name: 'lookup', arguments: input },
                      ],
                      stopReason: 'toolUse' as const,
                      timestamp: 0,
                      usage: {
                        input: 0,
                        output: 0,
                        cacheRead: 0,
                        cacheWrite: 0,
                        totalTokens: 0,
                        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
                      },
                    },
                    {
                      role: 'toolResult' as const,
                      toolCallId: 'call',
                      toolName: 'lookup',
                      content: [{ type: 'text' as const, text: '42' }],
                      isError: false,
                      timestamp: 0,
                    },
                  ]
                : []),
            ],
          }),
        )
        .result()
      if (response.stopReason === 'error') throw new Error(response.errorMessage)
      return
    }
    if (kind === 'strands' || kind === 'strands-browser') {
      const model =
        kind === 'strands'
          ? new WebLLMModel({ engine, modelId })
          : new BrowserModel(page, { modelId })
      await collect(
        model.stream([
          new Message({ role: 'user', content: [new TextBlock('Read a value.')] }),
          ...(input
            ? [
                new Message({
                  role: 'assistant',
                  content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input })],
                }),
                new Message({
                  role: 'user',
                  content: [
                    new ToolResultBlock({
                      toolUseId: 'call',
                      content: [new TextBlock('42')],
                      status: 'success',
                    }),
                  ],
                }),
              ]
            : []),
        ]),
      )
      return
    }
    const model =
      kind === 'langchain'
        ? new ChatWebLLM({ engine, modelId })
        : new BrowserChatWebLLM(page, { modelId })
    await model.invoke([
      new HumanMessage('Read a value.'),
      ...(input
        ? [
            new AIMessage({
              content: '',
              tool_calls: [{ id: 'call', name: 'lookup', args: input }],
            }),
            new ToolMessage({ tool_call_id: 'call', content: '42' }),
          ]
        : []),
    ])
  }
  return { invoke, create, close: () => Promise.all([strands.close(), langchain.close()]) }
}

for (const kind of clients) {
  it.each(['a>b', '>code', 'code>'])(
    `${kind} rejects an unrepresentable argument name %s before inference`,
    async (name) => {
      const model = client(kind)
      const input = { [name]: 42 }
      try {
        await expect(model.invoke(input)).rejects.toThrow('parameter names')
        expect(model.create).not.toHaveBeenCalled()
        expect(input).toEqual({ [name]: 42 })
        await model.invoke()
        expect(model.create).toHaveBeenCalledOnce()
      } finally {
        await model.close()
      }
    },
  )

  it(`${kind} preserves angle brackets inside nested JSON keys and string values`, async () => {
    const model = client(kind)
    const input = { payload: { 'a>b': 42 }, text: 'a > b' }
    const before = structuredClone(input)
    try {
      await model.invoke(input)
      const messages = model.create.mock.calls[0]?.[0].messages
      expect(messages?.find((message) => message.role === 'assistant')?.content).toContain(
        '"a>b": 42',
      )
      expect(messages?.find((message) => message.role === 'assistant')?.content).toContain('a > b')
      expect(input).toEqual(before)
    } finally {
      await model.close()
    }
  })
}
