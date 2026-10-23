import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { Type } from '@earendil-works/pi-ai'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import type { Completion } from '@mlc-ai/web-llm'
import { postInitAndCheckFieldsCompletion } from '@mlc-ai/web-llm'
import { BrowserModel, WebLLMModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import { gemmaChunks } from '../../shared/web-llm/gemma/stream.js'
import type { StreamChunk } from '../../shared/web-llm/stream/chunks.js'
import { collect, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
const example = 'Example: <|tool_call>call:lookup{code:<|"|>A7<|"|>}<tool_call|>'
const source =
  '<|channel>thought\nFirst.<channel|>' + example + '<|channel>thought\nSecond.<channel|> Done.'
const schema = Type.Object({ code: Type.String() })
const tool = { name: 'lookup', description: 'Look up a code.', inputSchema: schema }
const definition = {
  type: 'function',
  function: { name: tool.name, description: tool.description, parameters: schema },
}

afterEach(() => vi.unstubAllGlobals())

for (const client of ['strands', 'strands-browser', 'langchain', 'langchain-browser', 'pi']) {
  it.each(
    ['unbound', 'none', 'active'].flatMap((mode) =>
      [1, 3, source.length].map((size) => ({ mode, size })),
    ),
  )(
    `${client} preserves literal Gemma tool text with $mode tools in $size characters`,
    async ({ mode, size }) => {
      const literal = mode === 'active' ? 'Example: `<|tool_call>`' : example
      const value =
        '<|channel>thought\nFirst.<channel|>' +
        literal +
        '<|channel>thought\nSecond.<channel|> Done.'
      const { engine, create: chat } = setup()
      const native: Completion[] = Array.from(
        { length: Math.ceil(value.length / size) },
        (_, index) => ({
          id: 'native',
          model: modelId,
          object: 'text_completion',
          created: 0,
          choices: [
            {
              index: 0,
              text: value.slice(index * size, (index + 1) * size),
              finish_reason: (index + 1) * size >= value.length ? 'stop' : null,
            },
          ],
        }),
      )
      const create = vi.spyOn(engine.completions, 'create').mockImplementation(async (request) => {
        postInitAndCheckFieldsCompletion(request, modelId)
        return (async function* () {
          yield* native
        })()
      })
      const first = strandsBridge(engine)
      const second = langchainBridge(engine)
      vi.stubGlobal('window', { webllmBridge: first, webllmLangChainBridge: second })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const onChunk = vi.fn()
      const options = { modelId, onChunk, request: { extra_body: { enable_thinking: true } } }
      let text: string
      let reasoning: string
      try {
        if (client.startsWith('strands')) {
          const config = {
            ...options,
            request: {
              ...options.request,
              ...(mode === 'none' && { tool_choice: 'none' as const }),
            },
          }
          const model =
            client === 'strands'
              ? new WebLLMModel({ engine, ...config })
              : new BrowserModel(page, config)
          const output = await collect(
            model.streamAggregated(
              [new Message({ role: 'user', content: [new TextBlock('Show tool syntax.')] })],
              mode !== 'unbound' ? { toolSpecs: [tool] } : {},
            ),
          )
          text = output
            .filter((part) => part.type === 'textBlock')
            .map((part) => part.text)
            .join('')
          reasoning = output
            .filter((part) => part.type === 'reasoningBlock')
            .map((part) => part.text ?? '')
            .join('')
          expect(output.some((part) => part.type === 'toolUseBlock')).toBe(false)
        } else if (client === 'pi') {
          const provider = createWebLLMProvider({ engine, contextWindow: 4096, ...options })
          const result = await provider
            .stream(
              provider.getModels()[0]!,
              normalizeContext({
                messages: [{ role: 'user', content: 'Show tool syntax.', timestamp: 0 }],
                ...(mode !== 'unbound' && {
                  tools: [{ name: tool.name, description: tool.description, parameters: schema }],
                }),
              }),
              mode === 'none' ? { toolChoice: 'none' } : {},
            )
            .result()
          expect(result.stopReason).toBe('stop')
          text = result.content
            .filter((part) => part.type === 'text')
            .map((part) => part.text)
            .join('')
          reasoning = result.content
            .filter((part) => part.type === 'thinking')
            .map((part) => part.thinking)
            .join('')
          expect(result.content.some((part) => part.type === 'toolCall')).toBe(false)
        } else {
          const model =
            client === 'langchain'
              ? new ChatWebLLM({ engine, ...options })
              : new BrowserChatWebLLM(page, options)
          const result = await model.invoke('Show tool syntax.', {
            ...(mode !== 'unbound' && { tools: [definition] }),
            ...(mode === 'none' && { tool_choice: 'none' }),
          })
          text = result.text
          reasoning = result.contentBlocks
            .filter((part) => part.type === 'reasoning')
            .map((part) => part.reasoning)
            .join('')
          expect(result.tool_calls).toEqual([])
        }
        expect(text).toBe(literal + ' Done.')
        expect(reasoning).toBe('First.Second.')
        expect(onChunk.mock.calls.map(([value]) => value)).toEqual(native)
        expect(create).toHaveBeenCalledOnce()
        if (mode === 'active')
          expect(create.mock.calls[0]?.[0].prompt).toContain('<|tool>declaration:')
        else expect(create.mock.calls[0]?.[0].prompt).not.toContain('<|tool>declaration:')
        expect(chat).not.toHaveBeenCalled()
      } finally {
        await first.close()
        await second.close()
      }
    },
  )
}

it.each([
  { tail: '<|channel>thought\nUnfinished', error: 'ended inside a thinking block', unread: '' },
  { tail: '<|channel>answer\nOops', error: 'unknown thinking channel', unread: 'Oops' },
])(
  'still rejects malformed thinking after literal tool text: $tail',
  async ({ tail, error, unread }) => {
    const value = example + tail
    for (const size of [1, value.length]) {
      const native: Completion[] = Array.from(
        { length: Math.ceil(value.length / size) },
        (_, index) => ({
          id: 'native',
          model: modelId,
          object: 'text_completion',
          created: 0,
          choices: [
            {
              index: 0,
              text: value.slice(index * size, (index + 1) * size),
              finish_reason: (index + 1) * size >= value.length ? 'stop' : null,
            },
          ],
        }),
      )
      async function* input() {
        yield* native
      }
      const observed = vi.fn()
      const output: StreamChunk[] = []
      await expect(
        (async () => {
          for await (const chunk of gemmaChunks(input(), [], { onChunk: observed }))
            output.push(chunk)
        })(),
      ).rejects.toThrow(error)
      expect(output.flatMap((chunk) => chunk.calls)).toEqual([])
      const received = Math.ceil((value.length - unread.length) / size)
      expect(observed.mock.calls.map(([chunk]) => chunk)).toEqual(native.slice(0, received))
    }
  },
)
