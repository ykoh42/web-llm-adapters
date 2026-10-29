import { checkInstall } from '../../../scripts/check-install.ts'

await checkInstall({
  root: new URL('../', import.meta.url),
  peerImports: ['@earendil-works/pi-ai/models', '@earendil-works/pi-ai/api/simple-options'],
  forbiddenImports: ['@strands-agents/', '@langchain/'],
  compatibilityDependencies: [
    [
      '@earendil-works/pi-ai@1.0.2',
      '@earendil-works/pi-agent-core@1.0.2',
      '@mlc-ai/web-llm@0.2.79',
    ],
    [
      '@earendil-works/pi-ai@1.4.0',
      '@earendil-works/pi-agent-core@1.4.0',
      '@mlc-ai/web-llm@0.2.86',
    ],
  ],
  consumerDependencies: [
    '@earendil-works/pi-ai@1.0.2',
    '@earendil-works/pi-agent-core@1.0.2',
    '@mlc-ai/web-llm@0.2.79',
  ],
  source: `
import assert from 'node:assert/strict'
import { MLCEngine } from '@mlc-ai/web-llm'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { Agent } from '@earendil-works/pi-agent-core'
import { createWebLLMProvider, withEngine } from 'pi-ai-web-llm'
const engine = new MLCEngine({logLevel:'SILENT'})
assert.equal(await withEngine(engine, (current) => current === engine), true)
engine.chat.completions.create = (async () => (async function*(){yield {id:'test',object:'chat.completion.chunk',created:0,model:'test',choices:[{index:0,delta:{content:'Hello'},finish_reason:'stop'}]}})()) as unknown as typeof engine.chat.completions.create
const provider = createWebLLMProvider({engine,modelId:'test',contextWindow:8192})
const models = createModels()
models.setProvider(provider)
const model = provider.getModels()[0]!
const api: 'webllm' = model.api
assert.equal(api,'webllm')
const stream = models.stream(model,{messages:[{role:'user',content:'Hi',timestamp:0}]},{request:{seed:7}})
const events: string[] = []
for await (const event of stream) events.push(event.type)
assert.deepEqual(events,['start','text_start','text_delta','text_end','done'])
const response = await stream.result()
assert.equal(response.stopReason,'stop')
assert.deepEqual(response.content,[{type:'text',text:'Hello'}])
const any = Object.getOwnPropertyDescriptor(AbortSignal, 'any')!
Object.defineProperty(AbortSignal, 'any', { ...any, value: undefined })
try {
  const result = await models.completeSimple(model, {messages:[{role:'user',content:'Hi',timestamp:0}]}, {
    signal: new AbortController().signal, timeoutMs: 5000,
  })
  assert.equal(result.stopReason, 'stop')
} finally {
  Object.defineProperty(AbortSignal, 'any', any)
}
const agent = new Agent({initialState:{model},streamFn:models.streamSimple.bind(models)})
await agent.prompt('Hello')
assert.equal(agent.state.messages.at(-1)?.role,'assistant')

const toolContext = {
  messages: [{ role: 'user' as const, content: 'Look up A7', timestamp: 0 }],
  tools: [
    {
      name: 'lookup',
      description: 'Look up a code',
      parameters: Type.Object({ code: Type.String() }),
    },
  ],
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
engine.chat.completions.create = (async () =>
  streamToolCall('tool_calls')) as unknown as typeof engine.chat.completions.create
const toolStream = models.stream(model, toolContext)
const toolEvents: string[] = []
for await (const event of toolStream) toolEvents.push(event.type)
assert.deepEqual(toolEvents, ['start', 'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done'])
assert.equal((await toolStream.result()).stopReason, 'toolUse')

engine.chat.completions.create = (async () =>
  streamToolCall('length')) as unknown as typeof engine.chat.completions.create
const truncatedStream = models.stream(model, toolContext)
const truncatedEvents: string[] = []
for await (const event of truncatedStream) truncatedEvents.push(event.type)
assert.deepEqual(truncatedEvents, ['start', 'toolcall_start', 'toolcall_delta', 'done'])
assert.equal((await truncatedStream.result()).stopReason, 'length')

let inferenceCalls = 0
let toolExecutions = 0
const finalChunk = {
  ...toolChunk('stop'),
  choices: [{ index: 0, delta: { content: 'Recovered' }, finish_reason: 'stop' }],
}
const truncatedChunk = {
  ...toolChunk('length'),
  choices: [{
    index: 0,
    delta: {
      tool_calls: [1, 2].map(index => ({
        index, type: 'function', function: { name: 'lookup', arguments: '{"code":"A7"}' },
      })),
    },
    finish_reason: 'length',
  }],
}
engine.chat.completions.create = (async () => {
  const reply = ++inferenceCalls === 1 ? truncatedChunk : finalChunk
  return (async function* () { yield reply })()
}) as unknown as typeof engine.chat.completions.create
const truncatedAgent = new Agent({
  initialState: {
    model,
    tools: [{
      ...toolContext.tools[0]!,
      label: 'Lookup',
      execute: async () => {
        toolExecutions++
        return { content: [{ type: 'text' as const, text: 'A7' }], details: {} }
      },
    }],
  },
  streamFn: models.streamSimple.bind(models),
})
await truncatedAgent.prompt('Look up A7')
assert.equal(toolExecutions, 0)
assert.equal(inferenceCalls, 2)
const toolResults = truncatedAgent.state.messages.filter(message => message.role === 'toolResult')
assert.equal(toolResults.length, 2)
assert.ok(toolResults.every(result => result.isError && result.toolCallId))
assert.equal(new Set(toolResults.map(result => result.toolCallId)).size, 2)
const finalMessage = truncatedAgent.state.messages.at(-1)
assert.equal(finalMessage?.role, 'assistant')
if (finalMessage?.role === 'assistant') {
  assert.equal(finalMessage.stopReason, 'stop')
  assert.deepEqual(finalMessage.content, [{ type: 'text', text: 'Recovered' }])
}

engine.chat.completions.create = (async () => (async function* () {
  yield {
    id: 'test', object: 'chat.completion.chunk', created: 0, model: 'test',
    choices: [{ index: 0, delta: { content: 'Before<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>After' }, finish_reason: 'stop' }],
  }
})()) as unknown as typeof engine.chat.completions.create
const qwenProvider = createWebLLMProvider({
  engine, modelId: 'Qwen3-4B-q4f16_1-MLC', contextWindow: 8192, providerId: 'qwen',
})
models.setProvider(qwenProvider)
const qwen = qwenProvider.getModels()[0]!
const ordered = await models.complete(qwen, toolContext)
assert.equal(ordered.stopReason, 'toolUse')
assert.deepEqual(ordered.content.map(part => part.type), ['text', 'toolCall', 'text'])

let replay = ''
engine.chat.completions.create = (async (request: { messages: { role: string; content?: unknown }[] }) => {
  replay = String(request.messages.find(message => message.role === 'assistant')?.content)
  return (async function* () {
    yield { id: 'test', object: 'chat.completion.chunk', created: 0, model: 'test', choices: [{ index: 0, delta: { content: 'Done' }, finish_reason: 'stop' }] }
  })()
}) as unknown as typeof engine.chat.completions.create
const call = ordered.content.find(part => part.type === 'toolCall')
if (!call || call.type !== 'toolCall') throw new Error('Missing tool call.')
call.arguments = { value: -0 }
await models.complete(qwen, {
  messages: [
    ...toolContext.messages,
    ordered,
    { role: 'toolResult', toolCallId: call.id, toolName: call.name, content: [{ type: 'text', text: 'Done' }], isError: false, timestamp: 0 },
  ],
  tools: toolContext.tools,
})
assert(replay.includes('"value":-0'))

`,
})
