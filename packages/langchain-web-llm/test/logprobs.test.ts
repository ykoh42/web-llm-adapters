import { afterEach, expect, it, vi } from 'vitest'
import { AIMessage, AIMessageChunk, HumanMessage } from '@langchain/core/messages'
import type { ChatCompletionChunk, Completion } from '@mlc-ai/web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
  type WebLLMChunk,
} from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

const probabilities = ['A', 'B'].map((token, index) => ({
  token,
  bytes: index ? null : [token.charCodeAt(0)],
  logprob: index ? Number.NEGATIVE_INFINITY : -0.1,
  top_logprobs: [{ token, bytes: null, logprob: index ? Number.NEGATIVE_INFINITY : -0.1 }],
}))
const tools = [{ type: 'function', function: { name: 'lookup', parameters: { type: 'object' } } }]
const cases = [
  { modelId: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', text: 'Hello' },
  {
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    text: '<think>Plan.</think>Hello<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    text: '<think>Plan.</think>Hello<tool_call><function=lookup></function></tool_call>',
  },
  {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    text: '<|channel>thought\nPlan.<channel|>Hello<|tool_call>call:lookup{}<tool_call|>',
  },
]

afterEach(() => vi.unstubAllGlobals())

for (const browser of [false, true]) {
  it.each(
    cases.flatMap((test) =>
      ['invoke', 'stream', 'standard', 'events', 'generate'].map((mode) => ({ ...test, mode })),
    ),
  )(
    `preserves $modelId log probabilities through $mode (browser: ${browser})`,
    async ({ modelId, text, mode }) => {
      const raw = modelId.startsWith('gemma-')
      const first: WebLLMChunk = raw
        ? {
            id: 'native',
            object: 'text_completion',
            model: modelId,
            created: 0,
            choices: [{ index: 0, text, finish_reason: null }],
          }
        : chunk({ content: text })
      first.choices[0]!.logprobs = { content: [probabilities[0]!] }
      const last: WebLLMChunk = raw
        ? {
            ...first,
            object: 'text_completion',
            choices: [{ index: 0, text: '', finish_reason: 'stop' }],
          }
        : chunk({}, 'stop')
      last.choices[0]!.logprobs = { content: [probabilities[1]!] }
      const native = [first, last]
      const snapshot = structuredClone(native)
      const { engine } = setup(...(native as ChatCompletionChunk[]))
      vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
        (async function* () {
          yield* native as Completion[]
        })(),
      )
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmLangChainBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return structuredClone(result)
        },
      }
      const onChunk = vi.fn()
      const options = { modelId, onChunk, request: { logprobs: true, top_logprobs: 1 } }
      const model = browser
        ? new BrowserChatWebLLM(page, options)
        : new ChatWebLLM({ engine, ...options })
      const call = modelId.startsWith('Llama') ? {} : { tools }
      try {
        let result: unknown
        if (mode === 'events') {
          const stream = model.streamEvents('Hi', call)
          const events = []
          for await (const event of stream) events.push(event)
          expect(events.at(-1)).toMatchObject({
            event: 'message-finish',
            responseMetadata: { logprobs: { content: probabilities } },
          })
          expect(await stream.text).toBe('Hello')
          expect(await stream.reasoning).toBe(call.tools ? 'Plan.' : '')
          expect(await stream.toolCalls).toHaveLength(call.tools ? 1 : 0)
          result = await stream
        } else {
          if (mode === 'invoke') result = await model.invoke('Hi', call)
          else if (mode === 'generate') {
            const generation = (await model.generate([[new HumanMessage('Hi')]], call))
              .generations[0]![0]!
            expect(generation.generationInfo?.logprobs).toEqual({ content: probabilities })
            if (!('message' in generation)) throw new Error('Expected a chat generation.')
            result = generation.message
          } else {
            let message: AIMessageChunk | undefined
            for await (const part of await model.stream('Hi', {
              ...call,
              ...(mode === 'standard' && { outputVersion: 'v1' }),
            }))
              message = message ? message.concat(part) : part
            result = message!
          }
        }
        if (!AIMessage.isInstance(result)) throw new Error('Expected an AI response.')
        expect(result.text).toBe('Hello')
        expect(result.tool_calls).toHaveLength(call.tools ? 1 : 0)
        expect(result.response_metadata.logprobs).toEqual({ content: probabilities })
        expect(onChunk.mock.calls.map(([value]) => value)).toEqual(snapshot)
        expect(native).toEqual(snapshot)
      } finally {
        await bridge.close()
      }
    },
  )

  it.each([undefined, null, { content: null }, { content: [] }])(
    `preserves absent or empty log probabilities (browser: ${browser}): %j`,
    async (logprobs) => {
      const native = chunk({ content: 'Hello' }, 'stop')
      if (logprobs !== undefined) native.choices[0]!.logprobs = logprobs
      const { engine } = setup(native)
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmLangChainBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          return run(arg)
        },
      }
      const model = browser
        ? new BrowserChatWebLLM(page, { modelId: 'test-model' })
        : new ChatWebLLM({ engine, modelId: 'test-model' })
      try {
        const result = await model.invoke('Hi')
        expect(result.response_metadata.logprobs).toEqual(logprobs ?? undefined)
      } finally {
        await bridge.close()
      }
    },
  )
}
