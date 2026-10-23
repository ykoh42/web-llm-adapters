import { checkInstall } from '../../../scripts/check-install.ts'

await checkInstall({
  root: new URL('../', import.meta.url),
  peerImports: ['@langchain/core/messages'],
  forbiddenImports: ['@strands-agents/'],
  compatibilityDependencies: [
    ['@langchain/core@1.2.8', '@mlc-ai/web-llm@0.2.79'],
    ['@langchain/core@1.3.0', '@mlc-ai/web-llm@0.2.86'],
  ],
  consumerDependencies: ['@langchain/core@1.2.8', '@mlc-ai/web-llm@0.2.79'],
  source: `
import assert from 'node:assert/strict'
import { MLCEngine } from '@mlc-ai/web-llm'
import { ChatWebLLM, BrowserChatWebLLM, WebLLMEmbeddings, createBrowserBridge, withEngine } from 'langchain-web-llm'
const engine = new MLCEngine({logLevel:'SILENT'})
engine.chat.completions.create = (async () => (async function*(){yield {id:'test',object:'chat.completion.chunk',created:0,model:'test',choices:[{index:0,delta:{content:'Hello'},finish_reason:'stop'}]}})()) as unknown as typeof engine.chat.completions.create
const model = new ChatWebLLM({engine,modelId:'test'})
const response = await model.invoke('Hi')
assert.equal(response.text,'Hello')
const tokens: number | undefined = response.usage_metadata?.output_tokens
assert.equal(tokens,undefined)
const bound = await model.bindTools([]).invoke('Hi')
const boundTokens: number | undefined = bound.usage_metadata?.output_tokens
assert.equal(boundTokens,undefined)
for await (const chunk of await model.stream('Hi')) {
  const streamedTokens: number | undefined = chunk.usage_metadata?.output_tokens
  assert.equal(streamedTokens,undefined)
}
const batch = await model.batch(['One','Two'])
assert.equal(batch.length,2)
const batchTokens: number | undefined = batch[0]?.usage_metadata?.output_tokens
assert.equal(batchTokens,undefined)
assert.equal(typeof model.bindTools([]).invoke,'function')
assert.equal(typeof model.withStructuredOutput({type:'object'}).invoke,'function')
assert.equal(typeof new WebLLMEmbeddings({engine,modelId:'embedding'}).embedQuery,'function')
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
