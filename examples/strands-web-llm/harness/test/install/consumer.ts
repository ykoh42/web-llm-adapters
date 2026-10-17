import assert from 'node:assert/strict'
import { mock, test } from 'node:test'
import {
  MLCEngine,
  type ChatCompletionChunk,
  type ChatCompletionRequestBase,
} from '@mlc-ai/web-llm'
import { Agent, FunctionTool, Model, TextBlock } from '@strands-agents/sdk'
import { createHarness } from '@strands-agents/harness'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge,
  pcmAudio,
  withEngine,
  type BrowserPage,
} from 'strands-web-llm'

// Test the packed adapter with real SDK/Harness objects; inference is covered by GPU tests.
function open() {
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  const code = crypto.randomUUID()
  let calls = 0
  const tool = new FunctionTool({
    name: 'read_value',
    description: 'Read the stored value.',
    callback: () => {
      calls++
      return code
    },
  })
  mock.method(engine.chat.completions, 'create', async (request: ChatCompletionRequestBase) => {
    const result = request.messages.filter((message) => message.role === 'tool').at(-1)
    const chunk: ChatCompletionChunk = {
      id: 'install',
      model: 'test-model',
      object: 'chat.completion.chunk',
      created: 0,
      choices: [
        {
          index: 0,
          delta: result
            ? { content: result.content }
            : {
                tool_calls: [
                  { index: 0, id: 'read-1', function: { name: 'read_value', arguments: '{}' } },
                ],
              },
          finish_reason: result ? 'stop' : 'tool_calls',
        },
      ],
    }
    return (async function* () {
      yield chunk
    })()
  })
  return { engine, tool, code, calls: () => calls }
}

test('the packed provider shares the consumer SDK and completes an Agent tool loop', async () => {
  const { engine, tool, code, calls } = open()
  const model = new WebLLMModel({ engine, modelId: 'test-model' })
  assert.ok(model instanceof Model)
  assert.ok(
    import.meta
      .resolve('strands-web-llm')
      .startsWith(new URL('./node_modules/', import.meta.url).href),
  )
  assert.equal(await withEngine(engine, (current) => current === engine), true)
  assert.equal(pcmAudio(new Float32Array(16), 16000).type, 'audioBlock')
  const result = await new Agent({ model, tools: [tool], printer: false }).invoke('Read the value.')
  assert.deepEqual(result.lastMessage.content, [new TextBlock(code)])
  assert.equal(calls(), 1)
})

test('the packed browser provider is accepted by Harness and returns a tool result', async () => {
  const { engine, tool, code, calls } = open()
  const bridge = createBrowserBridge(engine)
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { webllmBridge: bridge },
  })
  const page: BrowserPage = {
    evaluate: async (run, arg) => {
      const result = await run(arg)
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  const model = new BrowserModel(page, { modelId: 'test-model', contextWindowLimit: 16384 })
  assert.ok(model instanceof Model)
  // Memory extraction is a separate model task, outside this installation check.
  const agent = await createHarness({
    model,
    auxModel: model,
    tools: [tool],
    memory: false,
    printer: false,
  })
  try {
    assert.equal(agent.model, model)
    const result = await agent.invoke('Read the value.', { limits: { turns: 3 } })
    assert.deepEqual(result.lastMessage.content, [new TextBlock(code)])
    assert.equal(calls(), 1)
  } finally {
    try {
      await agent.shutdown()
    } finally {
      await bridge.close()
      Reflect.deleteProperty(globalThis, 'window')
    }
  }
})
