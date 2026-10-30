import { checkInstall } from '../../../scripts/check-install.ts'

await checkInstall({
  root: new URL('../', import.meta.url),
  peerImports: ['@langchain/core/messages'],
  forbiddenImports: ['@strands-agents/', '@earendil-works/'],
  compatibilityDependencies: [
    ['@langchain/core@1.2.8', '@mlc-ai/web-llm@0.2.86'],
    ['@langchain/core@1.3.1', '@mlc-ai/web-llm@0.2.86'],
  ],
  consumerDependencies: ['@langchain/core@1.2.8', '@mlc-ai/web-llm@0.2.86'],
  source: `
import assert from 'node:assert/strict'
import { MLCEngine } from '@mlc-ai/web-llm'
import { CallbackManager } from '@langchain/core/callbacks/manager'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { ChatWebLLM, BrowserChatWebLLM, WebLLMEmbeddings, createBrowserBridge, withEngine } from 'langchain-web-llm'
const engine = new MLCEngine({logLevel:'SILENT'})
engine.chat.completions.create = (async () => (async function*(){yield {id:'test',object:'chat.completion.chunk',created:0,model:'test',choices:[{index:0,delta:{content:'Hello'},finish_reason:'stop'}]}})()) as unknown as typeof engine.chat.completions.create
const model = new ChatWebLLM({engine,modelId:'test'})
const penalty: number | null | undefined = model.invocationParams().repetition_penalty
assert.equal(penalty,undefined)
const identityPenalty: number | null | undefined = model._identifyingParams().repetition_penalty
assert.equal(identityPenalty,undefined)
const response = await model.invoke('Hi')
assert.equal(response.text,'Hello')
const signalled = await model.invoke('Hi', { signal: new AbortController().signal })
assert.equal(signalled.text,'Hello')
const complete = engine.chat.completions.create
const failure = Object.assign(new Error('Invalid request'), { status: 400 })
let attempts = 0
engine.chat.completions.create = async () => { attempts++; throw failure }
const retrying = new ChatWebLLM({ engine, modelId: 'test', maxRetries: 2 })
await assert.rejects(retrying.invoke('Hi', { signal: new AbortController().signal }), error => error === failure)
assert.equal(attempts,1)
engine.chat.completions.create = complete
const tokens: number | undefined = response.usage_metadata?.output_tokens
assert.equal(tokens,undefined)
const bound = await model.bindTools([]).invoke('Hi')
const boundTokens: number | undefined = bound.usage_metadata?.output_tokens
assert.equal(boundTokens,undefined)
for await (const chunk of await model.stream('Hi')) {
  const streamedTokens: number | undefined = chunk.usage_metadata?.output_tokens
  assert.equal(streamedTokens,undefined)
}
const events = model.streamEvents('Hi')
const eventNames: string[] = []
for await (const event of events) eventNames.push(event.event)
assert.equal(eventNames.at(-1),'message-finish')
assert(eventNames.includes('content-block-finish'))
const batch = await model.batch(['One','Two'])
assert.equal(batch.length,2)
const batchTokens: number | undefined = batch[0]?.usage_metadata?.output_tokens
assert.equal(batchTokens,undefined)
assert.equal(typeof model.bindTools([]).invoke,'function')
assert.equal(typeof model.withStructuredOutput({type:'object'}).invoke,'function')
assert.equal(typeof new WebLLMEmbeddings({engine,modelId:'embedding'}).embedQuery,'function')

const lookup = {
  type: 'function',
  function: {
    name: 'lookup',
    description: 'Look up a code.',
    parameters: {
      type: 'object',
      properties: { code: { type: 'string' } },
      required: ['code'],
    },
  },
}
const toolChunk = (finishReason: string) => ({
  id: 'test',
  object: 'chat.completion.chunk',
  created: 0,
  model: 'test',
  choices: [
    {
      index: 0,
      delta: {
        tool_calls: [
          {
            index: 0,
            id: 'call-1',
            type: 'function',
            function: { name: 'lookup', arguments: '{"code":"A7"}' },
          },
        ],
      },
      finish_reason: finishReason,
    },
  ],
})
const streamToolCall = (finishReason: string) =>
  (async function* () {
    yield toolChunk(finishReason)
  })()
const completeToolCall = async () => streamToolCall('tool_calls')
engine.chat.completions.create = completeToolCall as unknown as typeof engine.chat.completions.create
const toolStream = model.streamEvents('Look up A7.', { tools: [lookup] })
const completedCalls = []
for await (const call of toolStream.toolCalls) completedCalls.push(call)
assert.deepEqual(
  completedCalls.map((call) => call.args),
  [{ code: 'A7' }],
)

const truncatedToolCall = async () => streamToolCall('length')
engine.chat.completions.create = truncatedToolCall as unknown as typeof engine.chat.completions.create
const truncatedToolStream = model.streamEvents('Look up A7.', { tools: [lookup] })
const incompleteCalls = []
for await (const call of truncatedToolStream.toolCalls) incompleteCalls.push(call)
assert.deepEqual(incompleteCalls, [])
for (const preference of ['lc_prefer_streaming', 'lc_prefer_chat_model_stream_events']) {
  const callbacks = CallbackManager.fromHandlers({ handleLLMNewToken: () => {}, handleChatModelStreamEvent: () => {} })
  Object.assign(callbacks.handlers[0]!, { [preference]: true })
  const response = await model.bindTools([lookup]).invoke('Look up A7.', { callbacks })
  assert.deepEqual(response.tool_calls, [])
}

engine.chat.completions.create = (async () => (async function* () {
  yield {
    id: 'test', object: 'chat.completion.chunk', created: 0, model: 'test',
    choices: [{ index: 0, delta: { content: 'Before<tool_call>{"name":"lookup","arguments":{"code":"A7","value":-0.0}}</tool_call>After' }, finish_reason: 'stop' }],
  }
})()) as unknown as typeof engine.chat.completions.create
const qwen = new ChatWebLLM({ engine, modelId: 'Qwen3-4B-q4f16_1-MLC' })
const ordered = qwen.streamEvents('Look up A7.', { tools: [lookup] })
const blocks: string[] = []
for await (const event of ordered)
  if (event.event === 'content-block-start') blocks.push(event.content.type)
assert.deepEqual(blocks, ['text', 'tool_call_chunk', 'text'])
const args = (await ordered.toolCalls)[0]?.args as { value: unknown }
assert(Object.is(args.value, -0))

let replay = ''
engine.chat.completions.create = (async (request: { messages: { role: string; content?: unknown }[] }) => {
  replay = String(request.messages.find(message => message.role === 'assistant')?.content)
  return (async function* () {
    yield { id: 'test', object: 'chat.completion.chunk', created: 0, model: 'test', choices: [{ index: 0, delta: { content: 'Done' }, finish_reason: 'stop' }] }
  })()
}) as unknown as typeof engine.chat.completions.create
await qwen.invoke([
  new HumanMessage('Look up A7.'),
  new AIMessage({ content: '', tool_calls: [{ id: 'one', name: 'lookup', args: { value: -0 }, type: 'tool_call' }] }),
  new ToolMessage({ content: 'Done', tool_call_id: 'one' }),
])
assert(replay.includes('"value":-0'))

const bridge = createBrowserBridge(engine)
assert.equal(typeof bridge.start,'function')
const nativeValue: number = await bridge.withEngine(async native => {assert.equal(native,engine);return 42})
assert.equal(nativeValue,42)
const proxy = new BrowserChatWebLLM({evaluate:async () => {throw new Error('transport')}},{modelId:'test'})
assert.equal(proxy.modelId,'test')
await withEngine(engine, async native => assert.equal(native,engine))
await bridge.close()
`,
})
