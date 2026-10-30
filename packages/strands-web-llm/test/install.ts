import { checkInstall } from '../../../scripts/check-install.ts'

await checkInstall({
  root: new URL('../', import.meta.url),
  peerImports: ['@strands-agents/sdk'],
  forbiddenImports: ['@langchain/', '@earendil-works/'],
  compatibilityDependencies: [
    ['@strands-agents/sdk@1.18.0', '@mlc-ai/web-llm@0.2.86'],
    ['@strands-agents/sdk@1.22.0', '@mlc-ai/web-llm@0.2.86'],
  ],
  consumerDependencies: ['@strands-agents/sdk@1.18.0', '@mlc-ai/web-llm@0.2.86'],
  source: `
import assert from 'node:assert/strict'
import { runInNewContext } from 'node:vm'
import { Agent, AudioBlock, FunctionTool, ImageBlock, Message, TextBlock, ToolUseBlock, ToolResultBlock } from '@strands-agents/sdk'
import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { MLCEngine } from '@mlc-ai/web-llm'
import {
  BrowserModel,
  WebLLMEmbeddingSearchStrategy,
  WebLLMModel,
  createBrowserBridge,
  withEngine,
} from 'strands-web-llm'

const engine = new MLCEngine({ logLevel: 'SILENT' })
const chunk = (
  delta: Record<string, unknown>,
  finish: string,
) => ({
  id: 'test',
  model: 'test-model',
  object: 'chat.completion.chunk',
  created: 0,
  choices: [{ index: 0, delta, finish_reason: finish }],
})
let turn = 0
engine.chat.completions.create = (async () => {
  turn++
  const items = turn === 1
    ? [
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: 'call-1',
                type: 'function',
                function: { name: 'echo', arguments: '{}' },
              },
            ],
          },
          'tool_calls',
        ),
      ]
    : turn === 2
      ? [chunk({ content: 'Finished' }, 'stop')]
      : [
          chunk(
            {
              tool_calls: [
                {
                  index: 0,
                  id: 'call-2',
                  type: 'function',
                  function: { name: 'echo', arguments: '{' },
                },
              ],
            },
            'length',
          ),
        ]
  return (async function* () {
    yield* items
  })()
}) as unknown as typeof engine.chat.completions.create

let toolCalls = 0
const agent = new Agent({
  model: new WebLLMModel({ engine, modelId: 'test-model' }),
  tools: [
    new FunctionTool({
      name: 'echo',
      description: 'Echo text',
      callback: () => {
        toolCalls++
        return 'Echoed'
      },
    }),
  ],
  printer: false,
})
assert.match(String(await agent.invoke('Echo this')), /Finished/)
assert.equal(toolCalls, 1)
await assert.rejects(agent.invoke('Echo a truncated call'))
assert.equal(toolCalls, 1)

engine.chat.completions.create = (async () => (async function* () {
  yield chunk({ content: 'Before<tool_call>{"name":"echo","arguments":{}}</tool_call>After' }, 'stop')
})()) as unknown as typeof engine.chat.completions.create
const qwen = new WebLLMModel({ engine, modelId: 'Qwen3-4B-q4f16_1-MLC' })
const blocks: string[] = []
for await (const event of qwen.stream(
  [new Message({ role: 'user', content: [new TextBlock('Echo')] })],
  { toolSpecs: [{ name: 'echo', description: 'Echo text' }] },
)) {
  if (event.type === 'modelContentBlockStartEvent')
    blocks.push(event.start?.type === 'toolUseStart' ? 'tool' : 'text')
}
assert.deepEqual(blocks, ['text', 'tool', 'text'])

let replay = ''
engine.chat.completions.create = (async (request: { messages: { role: string; content?: unknown }[] }) => {
  replay = String(request.messages.find(message => message.role === 'assistant')?.content)
  return (async function* () { yield chunk({ content: 'Done' }, 'stop') })()
}) as unknown as typeof engine.chat.completions.create
const history = [
  new Message({ role: 'user', content: [new TextBlock('Echo')] }),
  new Message({ role: 'assistant', content: [new ToolUseBlock({ name: 'echo', toolUseId: 'one', input: { value: -0 } })] }),
  new Message({ role: 'user', content: [new ToolResultBlock({ toolUseId: 'one', status: 'success', content: [new TextBlock('Done')] })] }),
]
for await (const _ of qwen.stream(history)) {}
assert(replay.includes('"value":-0'))

const qwen35 = new WebLLMModel({ engine, modelId: 'Qwen3.5-4B-q4f16_1-MLC' })
const estimate = await qwen35.countTokens(
  [new Message({ role: 'user', content: [new TextBlock('Weather in Seoul?')] })],
  { toolSpecs: [{ name: 'weather', description: 'Get the weather.' }] },
)
assert(estimate > 200)

const qwen25 = new WebLLMModel({ engine, modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC' })
const reference = new WebLLMModel({ engine, modelId: 'test-model' })
const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]
const systemPrompt = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
assert.equal(await qwen25.countTokens(messages, {}), await reference.countTokens(messages, { systemPrompt }))
assert.equal(await qwen25.countTokens(messages), await reference.countTokens(messages))
const gemma = new WebLLMModel({
  engine,
  modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
  request: { extra_body: { enable_thinking: true } },
})
assert.equal(await gemma.countTokens(messages), await reference.countTokens(messages))
assert((await gemma.countTokens(messages, {})) > (await reference.countTokens(messages)))

const strategy = new WebLLMEmbeddingSearchStrategy({
  includeData: false,
  embed: async (texts) => texts.map(() => [1, 0]),
})
const storage = new InMemoryStorage(strategy)
await storage.write('note', new TextEncoder().encode('Cloud facts'))
const match = (await storage.search('cloud'))[0]
assert.equal(match?.key, 'note')
assert.equal(match?.data, undefined)
strategy.clearCache()
assert.equal((await storage.search('cloud'))[0]?.key, 'note')

const page = {
  async evaluate<T, A>(_run: (arg: A) => T | Promise<T>, _arg: A): Promise<T> {
    throw new Error('unused')
  },
}
assert.equal(new BrowserModel(page, { modelId: 'test-model' }).getConfig().modelId, 'test-model')

let sent = ''
const mediaPage = {
  async evaluate<T, A>(_run: (arg: A) => T | Promise<T>, arg: A): Promise<T> {
    sent = String(arg)
    throw new Error('request captured')
  },
}
const content = (bytes: Uint8Array) => [
  new ImageBlock({ format: 'png', source: { bytes } }),
  new AudioBlock({ format: 'wav', source: { bytes } }),
  new ToolResultBlock({ toolUseId: 'call', status: 'success', content: [
    new ImageBlock({ format: 'png', source: { bytes } }),
  ] }),
]
const foreignBytes = runInNewContext('new Uint8Array([0, 1, 2, 3, 4]).subarray(1, 4)') as Uint8Array
const media = new BrowserModel(mediaPage, { modelId: 'test-model' })
await assert.rejects(media.stream([new Message({ role: 'user', content: content(foreignBytes) })]).next(), /request captured/)
assert.deepEqual(JSON.parse(sent).messages[0].content, new Message({
  role: 'user', content: content(new Uint8Array([1, 2, 3])),
}).toJSON().content)

const bridge = createBrowserBridge(engine)
assert.equal(typeof bridge.start, 'function')
await bridge.close()
assert.equal(await withEngine(engine, (current) => current === engine), true)
`,
})
