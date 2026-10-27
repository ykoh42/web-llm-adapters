import { readFile } from 'node:fs/promises'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import {
  prebuiltAppConfig,
  type ChatCompletionChunk,
  type ChatCompletionRequestStreaming,
  type Completion,
  type CompletionCreateParamsStreaming,
  type MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { Message, TextBlock, ToolResultBlock } from '@strands-agents/sdk'
import { ChatWebLLM } from 'langchain-web-llm'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'
import type { Page } from '@playwright/test'
import { test, expect } from '../../examples/strands-web-llm/fixtures/browser.js'
import type {} from '../../examples/strands-web-llm/browser/src/harness/types.js'
import { qwenTurns } from '../support/qwen-output.js'

const sources = [
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
]
const schema = Type.Object({ code: Type.String() }, { additionalProperties: false })
const tool = { name: 'lookup', description: 'Look up a code.', inputSchema: schema }
const definition = {
  type: 'function',
  function: { name: tool.name, description: tool.description, parameters: schema },
}
const input = 'Call lookup with code A7. Then report the returned code exactly.'
const receipt = 'CODE-42'

async function read(stream: ReturnType<BrowserModel['streamAggregated']>) {
  for (;;) {
    const next = await stream.next()
    if (next.done) return next.value.message.content
  }
}

/** Forward real Worker output to the Node adapters without changing its chunks. */
function browserEngine(page: Page): MLCEngineInterface {
  function create(kind: 'chat' | 'completion') {
    return async (request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming) => {
      const result = await page.evaluate(
        async ({ kind, request }) =>
          window.webllmBridge.withEngine(async (engine) => {
            const stream =
              kind === 'chat'
                ? await engine.chat.completions.create(request as ChatCompletionRequestStreaming)
                : await engine.completions.create(request as CompletionCreateParamsStreaming)
            const chunks: (ChatCompletionChunk | Completion)[] = []
            for await (const item of stream) chunks.push(item)
            return chunks
          }),
        { kind, request },
      )
      return (async function* () {
        yield* result
      })()
    }
  }
  return {
    chat: { completions: { create: create('chat') } },
    completions: { create: create('completion') },
    interruptGenerate: () =>
      page.evaluate(() => window.webllmBridge.withEngine((engine) => engine.interruptGenerate())),
  } as unknown as MLCEngineInterface
}

type Options = {
  modelId: string
  sourceModelId: string
  thinkingPrefilled: boolean
  request: {
    max_tokens: number
    seed: number
    temperature: number
    top_p: number
    repetition_penalty: number
    constrainedToolCalls?: boolean
    presence_penalty?: number
    extra_body?: { enable_thinking: boolean }
  }
  onChunk: (chunk: WebLLMChunk) => void
}

async function strands(page: Page, options: Options) {
  const model = new BrowserModel(page, options)
  const messages = [new Message({ role: 'user', content: [new TextBlock(input)] })]
  const first = await read(
    model.streamAggregated(messages, {
      toolSpecs: [tool],
      toolChoice: { tool: { name: 'lookup' } },
    }),
  )
  const call = first.find((block) => block.type === 'toolUseBlock')
  if (!call) throw new Error('Missing Strands tool call.')
  expect(call.name).toBe('lookup')
  expect(call.input).toEqual({ code: 'A7' })
  const final = await read(
    model.streamAggregated(
      [
        ...messages,
        new Message({ role: 'assistant', content: first }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: call.toolUseId,
              status: 'success',
              content: [new TextBlock(receipt)],
            }),
          ],
        }),
      ],
      { toolSpecs: [tool] },
    ),
  )
  return {
    answer: final
      .filter((block) => block.type === 'textBlock')
      .map((block) => block.text)
      .join(''),
    thinking: [...first, ...final]
      .filter((block) => block.type === 'reasoningBlock')
      .map((block) => block.text ?? '')
      .join(''),
  }
}

async function langchain(page: Page, options: Options) {
  const model = new ChatWebLLM({ engine: browserEngine(page), ...options })
  const first = await model.invoke(input, { tools: [definition], tool_choice: 'lookup' })
  const call = first.tool_calls?.[0]
  if (!call?.id) throw new Error('Missing LangChain tool call.')
  expect(call.name).toBe('lookup')
  expect(call.args).toEqual({ code: 'A7' })
  const final = await model.invoke(
    [new HumanMessage(input), first, new ToolMessage({ tool_call_id: call.id, content: receipt })],
    { tools: [definition] },
  )
  return {
    answer: final.text,
    thinking: [...first.contentBlocks, ...final.contentBlocks]
      .filter((block) => block.type === 'reasoning')
      .map((block) => block.reasoning)
      .join(''),
  }
}

async function pi(page: Page, options: Options) {
  const provider = createWebLLMProvider({
    engine: browserEngine(page),
    contextWindow: 8192,
    ...options,
  })
  const models = createModels()
  models.setProvider(provider)
  const model = provider.getModels()[0]!
  const messages = [{ role: 'user' as const, content: input, timestamp: 0 }]
  const tools = [{ name: tool.name, description: tool.description, parameters: schema }]
  const first = await models.complete(
    model,
    { messages, tools },
    { toolChoice: { type: 'function', function: { name: 'lookup' } } },
  )
  if (first.stopReason === 'error') throw new Error(first.errorMessage)
  const call = first.content.find((part) => part.type === 'toolCall')
  if (!call) throw new Error('Missing Pi tool call.')
  expect(call.name).toBe('lookup')
  expect(call.arguments).toEqual({ code: 'A7' })
  const final = await models.complete(model, {
    messages: [
      ...messages,
      first,
      {
        role: 'toolResult',
        toolCallId: call.id,
        toolName: call.name,
        content: [{ type: 'text', text: receipt }],
        isError: false,
        timestamp: 0,
      },
    ],
    tools,
  })
  if (final.stopReason === 'error') throw new Error(final.errorMessage)
  return {
    answer: final.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join(''),
    thinking: [...first.content, ...final.content]
      .filter((part) => part.type === 'thinking')
      .map((part) => part.thinking)
      .join(''),
  }
}

for (const source of sources) {
  test(
    `${source}: custom ID keeps tool calls and thinking across adapters`,
    { tag: '@release' },
    async ({ page }) => {
      const modelId = source.startsWith('Qwen') ? 'my-qwen' : 'my-gemma'
      const found = prebuiltAppConfig.model_list.find((item) => item.model_id === source)
      if (!found) throw new Error('Missing WebLLM model record.')
      const record = { ...structuredClone(found), model_id: modelId }
      if (source.startsWith('gemma-')) {
        const assets: Record<string, { file: string }> = JSON.parse(
          await readFile('examples/strands-web-llm/browser/dist/.vite/manifest.json', 'utf8'),
        )
        const runtime: { integrity: string } = JSON.parse(
          await readFile('shared/runtime/manifest.json', 'utf8'),
        )
        const wasm = assets['../../../packages/strands-web-llm/dist/runtime/gemma-4-e2b.wasm']
        if (!wasm) throw new Error('Missing Gemma runtime.')
        record.model_lib = new URL(wasm.file, 'http://127.0.0.1:4173/').href
        record.integrity = { ...record.integrity, model_lib: runtime.integrity }
      }
      const chatOptions = {
        context_window_size: 8192,
        ...(source.startsWith('Qwen3.5-') && {
          conv_config: { role_empty_sep: '\n<think>\n', stop_token_ids: [248044, 248046] },
        }),
      }
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id, 8192), source)
      try {
        await page.evaluate(
          async ({ record, chatOptions }) =>
            window.webllmBridge.withEngine(async (engine) => {
              engine.setAppConfig({ model_list: [record], cacheBackend: 'cache' })
              await engine.reload(record.model_id, chatOptions)
            }),
          { record, chatOptions },
        )
        for (const run of [strands, langchain, pi]) {
          const native: WebLLMChunk[] = []
          const options: Options = {
            modelId,
            sourceModelId: source,
            thinkingPrefilled: source.startsWith('Qwen3.5-'),
            request: {
              max_tokens: 4096,
              seed: 17,
              temperature: source.startsWith('Qwen2.5-')
                ? 0.7
                : source.startsWith('Qwen3-')
                  ? 0.6
                  : 1,
              top_p: source.startsWith('Qwen2.5-') ? 0.8 : 0.95,
              repetition_penalty: source.startsWith('Qwen2.5-') ? 1.05 : 1,
              ...(source.startsWith('Qwen') && { constrainedToolCalls: true }),
              ...(source.startsWith('Qwen3.5-') && { presence_penalty: 1.5 }),
              ...(!source.startsWith('Qwen2.5-') && { extra_body: { enable_thinking: true } }),
            },
            onChunk: (chunk) => native.push(chunk),
          }
          const result = await run(page, options)
          expect(result.answer).toContain(receipt)
          expect(native.length).toBeGreaterThan(0)
          expect(native.every((chunk) => chunk.model === modelId)).toBe(true)
          if (source.startsWith('gemma-')) {
            const turns = new Map<string, string>()
            for (const chunk of native) {
              const choice = chunk.choices[0]
              if (choice && 'text' in choice)
                turns.set(chunk.id, (turns.get(chunk.id) ?? '') + choice.text)
            }
            const raw = [...turns.values()]
              .map((text, index) => (index === 1 ? '<|channel>thought\n' + text : text))
              .join('')
            const thoughts = [...raw.matchAll(/<\|channel>thought\n([\s\S]*?)<channel\|>/g)]
              .map((match) => match[1])
              .join('')
            expect(result.thinking).toBe(thoughts)
          } else {
            const raw = qwenTurns(native, options.thinkingPrefilled).join('')
            const thoughts = source.startsWith('Qwen2.5-')
              ? ''
              : [...raw.matchAll(/<think>([\s\S]*?)<\/think>/g)].map((match) => match[1]).join('')
            expect(result.thinking).toBe(thoughts)
          }
        }
      } finally {
        await page.evaluate(() => window.harness.close())
      }
    },
  )
}
