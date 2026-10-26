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

`,
})
