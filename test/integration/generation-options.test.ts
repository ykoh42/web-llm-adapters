import { afterEach, expect, it, vi } from 'vitest'
import type {
  ChatCompletionRequestBase,
  Completion,
  ResumableGenerationConfig,
} from '@mlc-ai/web-llm'
import { Agent, TextBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge as createStrandsBridge,
} from 'strands-web-llm'
import { chunk, setupEngine } from '../../fixtures/web-llm.js'

// A new native option must have an explicit coverage decision.
const request = {
  max_tokens: 64,
  temperature: 0.2,
  top_p: 0.9,
  frequency_penalty: 0.1,
  presence_penalty: 0.2,
  repetition_penalty: 1.1,
  stop: ['done', 'finished'],
  logit_bias: { '7': -2 },
  logprobs: true,
  top_logprobs: 2,
  seed: 7,
  response_format: { type: 'json_object', schema: '{"type":"object"}' },
  ignore_eos: false,
  extra_body: {
    enable_thinking: null,
    enable_latency_breakdown: true,
    resumable: {
      enabled: true,
      sessionId: 'native-session',
      durabilityMode: 'exact',
      strictPersistence: true,
      checkpointPrompt: true,
      checkpointIntervalTokens: 32,
      storeCheckpointLogits: true,
    } satisfies Required<ResumableGenerationConfig>,
  } satisfies Required<NonNullable<ChatCompletionRequestBase['extra_body']>>,
} satisfies Required<
  Omit<
    ChatCompletionRequestBase,
    'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'tool_choice' | 'n'
  >
>

afterEach(() => vi.unstubAllGlobals())

for (const client of ['langchain', 'browser', 'pi'] as const) {
  it.each(['defaults', 'call'] as const)(
    `${client} preserves every native generation option from %s`,
    async (source) => {
      const native = chunk({ content: '{}' }, 'stop')
      const { engine, create } = setupEngine(native)
      const onChunk = vi.fn()
      const options = {
        modelId: 'test-model',
        onChunk,
        request: source === 'defaults' ? request : { seed: 99, max_tokens: 1 },
      }
      const call = source === 'call' ? { request } : {}
      const input = 'Keep this input: 한글 <>&'
      const bridge = createBrowserBridge(engine)
      vi.stubGlobal('window', { webllmLangChainBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, arg) {
          const result = await run(structuredClone(arg))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      try {
        if (client === 'pi') {
          const provider = createWebLLMProvider({ engine, contextWindow: 8192, ...options })
          const result = await provider
            .stream(
              provider.getModels()[0]!,
              normalizeContext({ messages: [{ role: 'user', content: input, timestamp: 0 }] }),
              call,
            )
            .result()
          expect(result.stopReason).toBe('stop')
          expect(result.content).toEqual([{ type: 'text', text: '{}' }])
        } else {
          const model =
            client === 'browser'
              ? new BrowserChatWebLLM(page, options)
              : new ChatWebLLM({ engine, ...options })
          expect((await model.invoke(input, call)).text).toBe('{}')
        }
        expect(create).toHaveBeenCalledExactlyOnceWith({
          ...request,
          model: 'test-model',
          stream: true,
          stream_options: { include_usage: true },
          n: 1,
          messages: [{ role: 'user', content: input }],
        })
        expect(onChunk).toHaveBeenCalledExactlyOnceWith(native)
      } finally {
        await bridge.close()
      }
    },
  )
}

const modelIds = [
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
]

for (const client of [
  'langchain',
  'langchain-browser',
  'pi',
  'strands',
  'strands-browser',
] as const) {
  it.each(
    modelIds.flatMap((modelId) => ['defaults', 'override'].map((source) => ({ modelId, source }))),
  )(`${client} preserves generation options on $modelId ($source)`, async ({ modelId, source }) => {
    const raw = modelId.startsWith('gemma-')
    const selected = {
      ...request,
      response_format: { type: 'text' as const },
      extra_body: {
        ...request.extra_body,
        enable_thinking: modelId.startsWith('Qwen2.5-') ? null : true,
        resumable: { ...request.extra_body.resumable, enabled: false },
      },
    }
    const snapshot = structuredClone(selected)
    const native: Completion = {
      id: 'raw',
      object: 'text_completion',
      created: 0,
      model: modelId,
      choices: [{ index: 0, text: '{}', finish_reason: 'stop' }],
    }
    const chat = chunk({ content: '{}' }, 'stop')
    const { engine, create } = setupEngine(chat)
    const complete = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
      (async function* () {
        yield native
      })(),
    )
    const onChunk = vi.fn()
    const options = {
      modelId,
      request: source === 'defaults' ? selected : { seed: 99, max_tokens: 1 },
      onChunk,
    }
    const call = source === 'override' ? { request: selected } : {}
    const bridge = createBrowserBridge(engine)
    const strandsBridge = createStrandsBridge(engine)
    vi.stubGlobal('window', { webllmLangChainBridge: bridge, webllmBridge: strandsBridge })
    const page: BrowserPage = {
      async evaluate(run, arg) {
        const result = await run(structuredClone(arg))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    const input = 'Keep this input: 한글 <>&'
    const system = 'Caller system'
    try {
      if (client === 'pi') {
        const provider = createWebLLMProvider({ engine, contextWindow: 8192, ...options })
        const result = await provider
          .stream(
            provider.getModels()[0]!,
            normalizeContext({
              systemPrompt: system,
              messages: [{ role: 'user', content: input, timestamp: 0 }],
            }),
            call,
          )
          .result()
        expect(result).toMatchObject({
          stopReason: 'stop',
          content: [{ type: 'text', text: '{}' }],
        })
      } else if (client === 'strands' || client === 'strands-browser') {
        const model =
          client === 'strands-browser'
            ? new BrowserModel(page, options)
            : new WebLLMModel({ engine, ...options })
        if (source === 'override') model.updateConfig({ request: selected })
        const result = await new Agent({ model, systemPrompt: system, printer: false }).invoke(
          input,
        )
        expect(result.lastMessage.content).toEqual([new TextBlock('{}')])
      } else {
        const model =
          client === 'langchain-browser'
            ? new BrowserChatWebLLM(page, options)
            : new ChatWebLLM({ engine, ...options })
        const result = await model.invoke(
          [
            { role: 'system', content: system },
            { role: 'user', content: input },
          ],
          call,
        )
        expect(result.text).toBe('{}')
      }
      const sampling = {
        ...selected,
        model: modelId,
        stream: true,
        stream_options: { include_usage: true },
        n: 1,
      }
      if (raw) {
        // Gemma uses its prompt for thinking; raw completions only accept latency options.
        const expected: Omit<ChatCompletionRequestBase, 'messages'> = { ...sampling }
        delete expected.response_format
        expect(complete).toHaveBeenCalledExactlyOnceWith({
          ...expected,
          prompt: `<|turn>system\n<|think|>\n${system}<turn|>\n<|turn>user\n${input}<turn|>\n<|turn>model\n`,
          extra_body: { enable_latency_breakdown: true },
        })
        expect(create).not.toHaveBeenCalled()
      } else {
        expect(create).toHaveBeenCalledExactlyOnceWith({
          ...sampling,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: input },
          ],
        })
        expect(complete).not.toHaveBeenCalled()
      }
      expect(onChunk).toHaveBeenCalledExactlyOnceWith(raw ? native : chat)
      expect(selected).toEqual(snapshot)
    } finally {
      await bridge.close()
      await strandsBridge.close()
    }
  })
}
