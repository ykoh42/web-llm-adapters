import { checkInstall } from '../../../scripts/check-install.ts'

await checkInstall({
  root: new URL('../', import.meta.url),
  peerImports: ['@strands-agents/sdk'],
  forbiddenImports: ['@langchain/'],
  compatibilityDependencies: [
    ['@strands-agents/sdk@1.18.0', '@mlc-ai/web-llm@0.2.79'],
    ['@strands-agents/sdk@1.22.0', '@mlc-ai/web-llm@0.2.86'],
  ],
  consumerDependencies: ['@strands-agents/sdk@1.18.0', '@mlc-ai/web-llm@0.2.79'],
  source: `
import assert from 'node:assert/strict'
import { Agent, FunctionTool } from '@strands-agents/sdk'
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

const strategy = new WebLLMEmbeddingSearchStrategy({
  includeData: false,
  embed: async (texts) => texts.map(() => [1, 0]),
})
const storage = new InMemoryStorage(strategy)
await storage.write('note', new TextEncoder().encode('Cloud facts'))
const match = (await storage.search('cloud'))[0]
assert.equal(match?.key, 'note')
assert.equal(match?.data, undefined)

const page = {
  async evaluate<T, A>(_run: (arg: A) => T | Promise<T>, _arg: A): Promise<T> {
    throw new Error('unused')
  },
}
assert.equal(new BrowserModel(page, { modelId: 'test-model' }).getConfig().modelId, 'test-model')
const bridge = createBrowserBridge(engine)
assert.equal(typeof bridge.start, 'function')
await bridge.close()
assert.equal(await withEngine(engine, (current) => current === engine), true)
`,
})
