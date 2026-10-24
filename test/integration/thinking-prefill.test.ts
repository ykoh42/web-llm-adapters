import { afterEach, expect, it, vi } from 'vitest'
import {
  Message,
  TextBlock,
  MaxTokensError,
  type ContentBlock,
  type ModelStreamEvent,
} from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { BrowserModel, WebLLMModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import { createWebLLMProvider, type WebLLMRequest } from 'pi-ai-web-llm'
import { chunk, setupEngine } from '../../fixtures/web-llm.js'

const clients = ['langchain', 'langchain-browser', 'strands', 'strands-browser', 'pi'] as const
const models = ['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']
type Client = (typeof clients)[number]
interface Options {
  thinkingPrefilled?: boolean
  request?: WebLLMRequest
  finish?: ChatCompletionChunk.Choice['finish_reason']
  tools?: boolean
}

const lookup = {
  name: 'lookup',
  description: 'Read a code.',
  inputSchema: { type: 'object' as const, properties: { code: { type: 'string' as const } } },
}

afterEach(() => vi.unstubAllGlobals())

async function read(client: Client, modelId: string, source: string, options: Options = {}) {
  const parts = [source.slice(0, 2), source.slice(2, 9), source.slice(9)]
  const native = parts.map((content, index) =>
    chunk({ content }, index === parts.length - 1 ? (options.finish ?? 'stop') : null),
  )
  const { engine, create } = setupEngine(...native)
  const onChunk = vi.fn()
  const config = {
    modelId,
    request: {
      max_tokens: 4096,
      seed: 7,
      extra_body: { enable_thinking: true },
      ...options.request,
    },
    onChunk,
    ...(options.thinkingPrefilled !== undefined && {
      thinkingPrefilled: options.thinkingPrefilled,
    }),
  }
  const first = langchainBridge(engine)
  const second = strandsBridge(engine)
  vi.stubGlobal('window', { webllmLangChainBridge: first, webllmBridge: second })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  let text: string
  let thinking: string
  let tools: { name: string; input: unknown }[]
  let error: MaxTokensError | undefined
  try {
    if (client === 'pi') {
      const provider = createWebLLMProvider({ engine, contextWindow: 32768, ...config })
      const message = await provider
        .stream(
          provider.getModels()[0]!,
          normalizeContext({
            messages: [{ role: 'user', content: 'Keep this input.', timestamp: 0 }],
            ...(options.tools && {
              tools: [
                {
                  name: lookup.name,
                  description: lookup.description,
                  parameters: lookup.inputSchema,
                },
              ],
            }),
          }),
        )
        .result()
      if (message.stopReason === 'error') throw new Error(message.errorMessage)
      text = message.content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('')
      thinking = message.content
        .filter((part) => part.type === 'thinking')
        .map((part) => part.thinking)
        .join('')
      tools = message.content
        .filter((part) => part.type === 'toolCall')
        .map((part) => ({ name: part.name, input: part.arguments }))
    } else if (client.startsWith('strands')) {
      const model =
        client === 'strands'
          ? new WebLLMModel({ engine, ...config })
          : new BrowserModel(page, config)
      const blocks: (ContentBlock | ModelStreamEvent)[] = []
      try {
        for await (const block of model.streamAggregated(
          [new Message({ role: 'user', content: [new TextBlock('Keep this input.')] })],
          options.tools ? { toolSpecs: [lookup] } : {},
        ))
          blocks.push(block)
      } catch (failure) {
        if (!(failure instanceof MaxTokensError)) throw failure
        error = failure
      }
      text = blocks
        .filter((part) => part.type === 'textBlock')
        .map((part) => part.text)
        .join('')
      thinking = blocks
        .filter((part) => part.type === 'reasoningBlock')
        .map((part) => part.text ?? '')
        .join('')
      tools = blocks
        .filter((part) => part.type === 'toolUseBlock')
        .map((part) => ({ name: part.name, input: part.input }))
    } else {
      const model =
        client === 'langchain'
          ? new ChatWebLLM({ engine, ...config })
          : new BrowserChatWebLLM(page, config)
      const message = await model.invoke(
        'Keep this input.',
        options.tools
          ? {
              tools: [
                {
                  type: 'function',
                  function: {
                    name: lookup.name,
                    description: lookup.description,
                    parameters: lookup.inputSchema,
                  },
                },
              ],
            }
          : {},
      )
      text = message.text
      thinking = message.contentBlocks
        .filter((part) => part.type === 'reasoning')
        .map((part) => part.reasoning)
        .join('')
      tools = (message.tool_calls ?? []).map((part) => ({ name: part.name, input: part.args }))
    }
    expect(create).toHaveBeenCalledOnce()
    expect(onChunk.mock.calls.map(([value]) => value)).toEqual(native)
    const request = create.mock.calls[0]![0]
    expect(request).not.toHaveProperty('thinkingPrefilled')
    return { text, thinking, tools, request, error }
  } finally {
    await first.close()
    await second.close()
  }
}

for (const client of clients) {
  it.each(models)(
    `${client} separates a declared prefilled thought on %s without changing native input`,
    async (modelId) => {
      const source = 'Plan.\n</think>\n\n42'
      const omitted = await read(client, modelId, source)
      const disabled = await read(client, modelId, source, { thinkingPrefilled: false })
      const enabled = await read(client, modelId, source, { thinkingPrefilled: true })
      expect(omitted.text).toBe(source)
      expect(omitted.thinking).toBe('')
      expect(disabled).toEqual(omitted)
      expect(enabled.text).toBe('\n\n42')
      expect(enabled.thinking).toBe('Plan.\n')
      expect(enabled.request).toEqual(omitted.request)
    },
  )

  it.each(models)(
    `${client} follows native thinking=false on a prefilled %s engine`,
    async (modelId) => {
      const result = await read(client, modelId, '<think>\n\n</think>\n\n42', {
        thinkingPrefilled: true,
        request: { extra_body: { enable_thinking: false } },
      })
      expect(result.thinking).toBe('\n\n')
      expect(result.text).toBe('\n\n42')
    },
  )

  it.each(models)(`${client} preserves native JSON on a prefilled %s engine`, async (modelId) => {
    const result = await read(client, modelId, '{"answer":42}', {
      thinkingPrefilled: true,
      request: { response_format: { type: 'json_object' } },
    })
    expect(result.text).toBe('{"answer":42}')
    expect(result.thinking).toBe('')
  })

  it.each(models)(
    `${client} keeps partial prefilled %s thinking on a token limit`,
    async (modelId) => {
      const result = await read(client, modelId, 'Still planning.', {
        thinkingPrefilled: true,
        finish: 'length',
      })
      if (client.startsWith('strands')) expect(result.error).toBeInstanceOf(MaxTokensError)
      else expect(result.error).toBeUndefined()
      expect(result.thinking).toBe('Still planning.')
      expect(result.text).toBe('')
      expect(result.tools).toEqual([])
    },
  )

  it.each(models)(`${client} honors caller stops inside prefilled %s thinking`, async (modelId) => {
    const result = await read(client, modelId, 'Plan.', {
      thinkingPrefilled: true,
      request: { stop: ['CUT'] },
    })
    expect(result.thinking).toBe('Plan.')
    expect(result.text).toBe('')
  })

  it.each(models)(`${client} reads a tool call after prefilled %s thinking`, async (modelId) => {
    const call = modelId.startsWith('Qwen3.5-')
      ? '<tool_call><function=lookup><parameter=code>A7</parameter></function></tool_call>'
      : '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
    const result = await read(client, modelId, 'Need a code.\n</think>' + call, {
      thinkingPrefilled: true,
      tools: true,
    })
    expect(result.thinking).toBe('Need a code.\n')
    expect(result.text).toBe('')
    expect(result.tools).toEqual([{ name: 'lookup', input: { code: 'A7' } }])
  })
}

const noPage: BrowserPage = {
  async evaluate() {
    throw new Error('No browser call expected.')
  },
}

function construct(client: Client, config: { modelId: string; thinkingPrefilled?: boolean }) {
  const { engine, create } = setupEngine()
  const build = () => {
    if (client === 'langchain') return new ChatWebLLM({ engine, ...config })
    if (client === 'langchain-browser') return new BrowserChatWebLLM(noPage, config)
    if (client === 'strands') return new WebLLMModel({ engine, ...config })
    if (client === 'strands-browser') return new BrowserModel(noPage, config)
    return createWebLLMProvider({ engine, contextWindow: 32768, ...config })
  }
  return { build, create }
}

for (const client of clients) {
  it.each([null, 1, 'true'])(`${client} rejects a non-boolean prefill flag: %s`, (value) => {
    const config = { modelId: models[0]!, thinkingPrefilled: true }
    Object.assign(config, { thinkingPrefilled: value })
    const { build, create } = construct(client, config)
    expect(build).toThrow('thinkingPrefilled must be a boolean.')
    expect(create).not.toHaveBeenCalled()
  })

  it.each(['test-model', 'Qwen2.5-3B-Instruct-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC'])(
    `${client} accepts a disabled prefill flag and rejects enabling it on %s`,
    (modelId) => {
      const disabled = construct(client, { modelId, thinkingPrefilled: false })
      expect(disabled.build).not.toThrow()
      const enabled = construct(client, { modelId, thinkingPrefilled: true })
      expect(enabled.build).toThrow('thinkingPrefilled requires a Qwen thinking model.')
      expect(enabled.create).not.toHaveBeenCalled()
    },
  )
}

it.each([false, true])(
  'keeps Strands config updates atomic when prefill metadata is invalid (browser=%s)',
  (browser) => {
    const { engine } = setupEngine()
    const config = { modelId: models[0]!, thinkingPrefilled: true }
    const model = browser
      ? new BrowserModel(noPage, config)
      : new WebLLMModel({ engine, ...config })
    const before = model.getConfig()
    expect(() => model.updateConfig({ modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })).toThrow(
      'thinkingPrefilled requires a Qwen thinking model.',
    )
    expect(model.getConfig()).toEqual(before)
    model.updateConfig({ modelId: 'gemma-4-E2B-it-q4f16_1-MLC', thinkingPrefilled: false })
    expect(model.getConfig()).toMatchObject({
      modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
      thinkingPrefilled: false,
    })
  },
)
