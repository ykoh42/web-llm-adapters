import { afterEach, expect, it, vi } from 'vitest'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import {
  ChatWebLLM,
  BrowserChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { WebLLMModel, BrowserModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import type { Completion } from '@mlc-ai/web-llm'
import { nativeChunks } from '../../shared/web-llm/native.js'
import type { JSONValue } from '../../shared/web-llm/types.js'
import { chunk, chunks, collect, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

const models = ['Hermes-3-Llama-3.1-8B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']
const clients = ['langchain', 'langchain-browser', 'strands', 'strands-browser', 'pi'] as const

function argumentsAtDepth(array: boolean, reordered: boolean, different = false) {
  let value: JSONValue = different ? 43 : 42
  const depth = array ? 1500 : 3000
  for (let level = 0; level < depth; level++)
    value = array ? [value] : reordered ? { level, next: value } : { next: value, level }
  return reordered ? { code: 'A7', tree: value } : { tree: value, code: 'A7' }
}

for (const array of [false, true]) {
  for (const different of [false, true]) {
    const input = argumentsAtDepth(array, true, different)
    const raw = JSON.stringify([
      { name: 'lookup', arguments: argumentsAtDepth(array, false), note: 'Keep this note.' },
    ])
    const expected = different ? raw + JSON.stringify([{ name: 'lookup', arguments: input }]) : raw

    for (const modelId of models) {
      it.each(clients)(
        `%s preserves deep ${modelId} history (array=${array}, different=${different})`,
        async (client) => {
          const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
          const gemma = modelId.startsWith('gemma-')
          const content = gemma ? '' : raw
          async function* response(): AsyncGenerator<Completion> {
            yield {
              id: 'test',
              model: modelId,
              object: 'text_completion',
              created: 0,
              choices: [{ index: 0, text: 'Done', finish_reason: 'stop' }],
            }
          }
          const complete = vi
            .spyOn(engine.completions, 'create')
            .mockImplementation(async () => response())
          const first = langchainBridge(engine)
          const second = strandsBridge(engine)
          vi.stubGlobal('window', { webllmLangChainBridge: first, webllmBridge: second })
          const page: BrowserPage = {
            async evaluate(run, arg) {
              const result = await run(structuredClone(arg))
              return result === undefined ? result : JSON.parse(JSON.stringify(result))
            },
          }
          try {
            if (client.startsWith('langchain')) {
              const model =
                client === 'langchain'
                  ? new ChatWebLLM({ engine, modelId })
                  : new BrowserChatWebLLM(page, { modelId })
              await model.invoke([
                new HumanMessage('Use lookup.'),
                new AIMessage({
                  content,
                  tool_calls: [{ id: 'call', name: 'lookup', args: input }],
                }),
                new ToolMessage({ tool_call_id: 'call', content: 'Done.' }),
              ])
            } else if (client.startsWith('strands')) {
              const model =
                client === 'strands'
                  ? new WebLLMModel({ engine, modelId })
                  : new BrowserModel(page, { modelId })
              await collect(
                model.stream([
                  new Message({ role: 'user', content: [new TextBlock('Use lookup.')] }),
                  new Message({
                    role: 'assistant',
                    content: [
                      new TextBlock(content),
                      new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input }),
                    ],
                  }),
                  new Message({
                    role: 'user',
                    content: [
                      new ToolResultBlock({
                        toolUseId: 'call',
                        status: 'success',
                        content: [new TextBlock('Done.')],
                      }),
                    ],
                  }),
                ]),
              )
            } else {
              const provider = createWebLLMProvider({ engine, modelId, contextWindow: 32768 })
              const model = provider.getModels()[0]!
              const result = await provider
                .stream(
                  model,
                  normalizeContext({
                    messages: [
                      { role: 'user', content: 'Use lookup.', timestamp: 0 },
                      {
                        role: 'assistant',
                        api: model.api,
                        provider: model.provider,
                        model: model.id,
                        content: [
                          { type: 'text', text: content },
                          { type: 'toolCall', id: 'call', name: 'lookup', arguments: input },
                        ],
                        stopReason: 'toolUse',
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
                        role: 'toolResult',
                        toolCallId: 'call',
                        toolName: 'lookup',
                        content: [{ type: 'text', text: 'Done.' }],
                        isError: false,
                        timestamp: 0,
                      },
                    ],
                  }),
                )
                .result()
              expect(result.stopReason).toBe('stop')
            }
            if (gemma) {
              expect(create).not.toHaveBeenCalled()
              expect(complete).toHaveBeenCalledOnce()
              let literal = different ? '43' : '42'
              for (let level = 0; level < (array ? 1500 : 3000); level++)
                literal = array ? '[' + literal + ']' : `{level:${level},next:${literal}}`
              const expectedPrompt =
                '<|turn>system\n<turn|>\n<|turn>user\nUse lookup.<turn|>\n<|turn>model\n' +
                '<|tool_call>call:lookup{code:<|"|>A7<|"|>,tree:' +
                literal +
                '}<tool_call|>' +
                '<|tool_response>response:lookup{value:<|"|>Done.<|"|>}<tool_response|>'
              const prompt = complete.mock.calls[0]?.[0].prompt
              expect(prompt?.length).toBe(expectedPrompt.length)
              expect(prompt === expectedPrompt).toBe(true)
            } else {
              expect(complete).not.toHaveBeenCalled()
              expect(create).toHaveBeenCalledOnce()
              const prepared = create.mock.calls[0]?.[0].messages[1]?.content
              expect(prepared?.length).toBe(expected.length)
              expect(prepared === expected).toBe(true)
            }
          } finally {
            await first.close()
            await second.close()
          }
        },
      )
    }

    it(`keeps deep native tool output distinct from text (array=${array}, different=${different})`, async () => {
      const source = JSON.stringify([{ name: 'lookup', arguments: argumentsAtDepth(array, false) }])
      const values = await collect(
        nativeChunks(
          chunks(
            chunk({ content: source }),
            chunk(
              {
                tool_calls: [
                  {
                    index: 0,
                    id: 'call',
                    function: { name: 'lookup', arguments: JSON.stringify(input) },
                  },
                ],
              },
              'tool_calls',
            ),
          ),
          true,
        ),
      )
      expect(values.map((part) => part.text).join('') === (different ? source : '')).toBe(true)
      const calls = values.flatMap((part) => part.calls)
      expect(calls).toHaveLength(1)
      expect(calls[0]?.input === JSON.stringify(input)).toBe(true)
    })
  }
}
