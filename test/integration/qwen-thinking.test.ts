import { afterEach, describe, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
} from '../../packages/langchain-web-llm/src/index.js'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge as strandsBridge,
  type BrowserPage,
} from '../../packages/strands-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, chunks, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'
import type { WebLLMRequest } from '../../shared/web-llm/request.js'

const clients = ['strands', 'strands browser', 'langchain', 'langchain browser', 'pi'] as const
const models = ['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']
const thought = 'Plan 서울 😀</thi'
const stops: WebLLMRequest['stop'][] = [undefined, null, '', [], 'DONE', ['DONE']]

interface Output {
  text: string
  reasoning: string
  stop: string | undefined
  error: unknown
}

afterEach(() => vi.unstubAllGlobals())

describe.each(clients)('%s Qwen thinking', (client) => {
  describe.each(models)('%s', (modelId) => {
    it.each(stops.map((stop) => ({ stop })))(
      'honors native stop $stop without accepting malformed completion',
      async ({ stop }) => {
        const raw = chunk({ content: '<think>' + thought }, 'stop')
        const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
        create.mockResolvedValueOnce(chunks(raw))
        const observed = vi.fn()
        const request: WebLLMRequest = {
          extra_body: { enable_thinking: true },
          ...(stop !== undefined && { stop }),
        }
        const saved = structuredClone(request)
        const config = { modelId, request, onChunk: observed }
        const local = new WebLLMModel({ engine, ...config })
        const chat = new ChatWebLLM({ engine, ...config })
        const first = strandsBridge(engine)
        const second = langchainBridge(engine)
        vi.stubGlobal('window', { webllmBridge: first, webllmLangChainBridge: second })
        const page: BrowserPage = {
          async evaluate(run, arg) {
            const value = await run(structuredClone(arg))
            return value === undefined ? value : JSON.parse(JSON.stringify(value))
          },
        }
        const browser = new BrowserModel(page, config)
        const browserChat = new BrowserChatWebLLM(page, config)
        const provider = createWebLLMProvider({ engine, contextWindow: 8192, ...config })

        async function run(): Promise<Output> {
          const result: Output = { text: '', reasoning: '', stop: undefined, error: undefined }
          try {
            if (client === 'strands' || client === 'strands browser') {
              const model = client === 'strands' ? local : browser
              for await (const event of model.stream([
                new Message({ role: 'user', content: [new TextBlock('Hi')] }),
              ])) {
                if (event.type === 'modelContentBlockDeltaEvent') {
                  if (event.delta.type === 'reasoningContentDelta')
                    result.reasoning += event.delta.text
                  if (event.delta.type === 'textDelta') result.text += event.delta.text
                }
                if (event.type === 'modelMessageStopEvent') result.stop = event.stopReason
              }
            } else if (client === 'pi') {
              const stream = provider.stream(
                provider.getModels()[0]!,
                normalizeContext({ messages: [{ role: 'user', content: 'Hi', timestamp: 0 }] }),
              )
              for await (const event of stream) {
                if (event.type === 'thinking_delta') result.reasoning += event.delta
                if (event.type === 'text_delta') result.text += event.delta
                if (event.type === 'done') result.stop = event.reason
                if (event.type === 'error') {
                  result.stop = event.reason
                  result.error = new Error(event.error.errorMessage)
                }
              }
            } else {
              const model = client === 'langchain' ? chat : browserChat
              for await (const message of await model.stream('Hi')) {
                result.text += message.text
                for (const block of message.contentBlocks)
                  if (block.type === 'reasoning') result.reasoning += block.reasoning
                const finish = message.response_metadata.finish_reason
                if (typeof finish === 'string') result.stop = finish
              }
            }
          } catch (error) {
            result.error = error
          }
          return result
        }

        try {
          const result = await run()
          expect(result.text).toBe('')
          expect(result.reasoning).toBe(thought)
          expect(observed).toHaveBeenCalledExactlyOnceWith(raw)
          expect(create.mock.calls[0]?.[0]).toMatchObject({
            model: modelId,
            messages: [{ role: 'user', content: 'Hi' }],
            extra_body: { enable_thinking: true },
          })
          expect(create.mock.calls[0]?.[0].stop).toEqual(stop)
          if (stop?.length) {
            expect(result.error).toBeUndefined()
            expect(result.stop).toBe(client.startsWith('strands') ? 'endTurn' : 'stop')
          } else {
            expect(result.error).toHaveProperty('message', 'Qwen ended inside a thinking block.')
            expect(result.stop).toBe(client === 'pi' ? 'error' : undefined)
          }
          expect(await run()).toMatchObject({ text: 'Recovered', reasoning: '', error: undefined })
          expect(request).toEqual(saved)
          expect(create).toHaveBeenCalledTimes(2)
        } finally {
          await first.close()
          await second.close()
        }
      },
    )
  })
})
