import process from 'node:process'
import { beforeAll, afterAll, expect, test } from 'vitest'
import { chromium } from 'playwright'
import { spawn } from 'node:child_process'
import { setTimeout } from 'node:timers/promises'
import { AIMessageChunk, AIMessage } from '@langchain/core/messages'
import { ChatModelIntegrationTests } from '../../../../.cache/langchain-standard/integration_tests/chat_models.js'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { BrowserChatWebLLM } from '../../src/index.js'

let context
let page
let server
class BrowserModel extends BrowserChatWebLLM {
  constructor(options) {
    super(page, options)
  }
}
class IntegrationTests extends ChatModelIntegrationTests {
  // Current core emits AIMessageChunk objects in v2 events, rather than text arrays.
  async testStreamEvents() {
    const model = new this.Cls(this.constructorArgs)
    const events = []
    for await (const event of model.streamEvents('Hello', { version: 'v2' })) events.push(event)
    expect(events[0].event).toBe('on_chat_model_start')
    expect(events.at(-1).event).toBe('on_chat_model_end')
    const output = events.at(-1).data.output
    expect(AIMessage.isInstance(output)).toBe(true)
    expect(output.text.length).toBeGreaterThan(0)
    const tokens = events
      .filter((event) => event.event === 'on_chat_model_stream')
      .map((event) => event.data.chunk.text)
      .join('')
    expect(tokens).toBe(output.text)
  }
  // WebLLM has no native call IDs. The adapter generates unique IDs, not a fixed fixture ID.
  async testToolCalling() {
    const model = new this.Cls(this.constructorArgs)
    const add = tool(({ a, b }) => String(a + b), {
      name: 'AdderTool',
      description: 'Add two integers',
      schema: z.object({ a: z.number().int(), b: z.number().int() }),
    })
    const result = await model
      .bindTools([add])
      .invoke('What is the value of adderTool(1, 2)? Use the tool.')
    expect(result.tool_calls).toHaveLength(1)
    expect(result.tool_calls[0]).toMatchObject({
      name: 'AdderTool',
      args: { a: 1, b: 2 },
      type: 'tool_call',
    })
    expect(result.tool_calls[0].id).toMatch(/^[0-9a-f-]{36}$/)
  }
}
const suite = new IntegrationTests({
  Cls: BrowserModel,
  constructorArgs: {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    request: {
      max_tokens: 4096,
      temperature: 1,
      top_p: 0.95,
      repetition_penalty: 1,
      seed: 7,
      extra_body: { enable_thinking: true },
    },
  },
  chatModelHasToolCalling: true,
  chatModelHasStructuredOutput: true,
  invokeResponseType: AIMessageChunk,
  functionId: 'call-test',
  supportsParallelToolCalls: true,
  supportsStandardContentType: { text: true },
})
suite.expect = expect

beforeAll(async () => {
  try {
    await globalThis.fetch('http://127.0.0.1:4173/langchain/')
  } catch {
    server = spawn('node', ['shared/browser/server.ts'], { cwd: process.cwd(), stdio: 'ignore' })
    const deadline = Date.now() + 30000
    while (true) {
      try {
        await globalThis.fetch('http://127.0.0.1:4173/langchain/')
        break
      } catch (error) {
        if (Date.now() > deadline) throw error
        await setTimeout(100)
      }
    }
  }
  context = await chromium.launchPersistentContext('.cache/chrome', {
    channel: 'chrome',
    headless: process.env.HEADLESS === '1',
  })
  page = context.pages()[0] ?? (await context.newPage())
  await page.goto('http://127.0.0.1:4173/langchain/')
  await page.evaluate(() => globalThis.window.checks.load('gemma-4-E2B-it-q4f16_1-MLC'))
}, 900000)
afterAll(async () => {
  await context?.close()
  server?.kill()
})

for (const method of [
  'testInvoke',
  'testStream',
  'testBatch',
  'testStreamEvents',
  'testConversation',
  'testDoubleMessageConversation',
  'testUsageMetadata',
  'testUsageMetadataStreaming',
  'testStopSequence',
  'testToolCalling',
  'testBindToolsWithRunnableToolLike',
  'testBindToolsWithOpenAIFormattedTools',
  'testToolMessageHistoriesStringContent',
  'testToolMessageHistoriesListContent',
  'testStructuredFewShotExamples',
  'testWithStructuredOutput',
  'testWithStructuredOutputIncludeRaw',
  'testCacheComplexMessageTypes',
  'testStreamTokensWithToolCalls',
  'testModelCanUseToolUseAIMessage',
  'testModelCanUseToolUseAIMessageWithStreaming',
  'testInvokeMoreComplexTools',
  'testParallelToolCalling',
  'testModelCanAcceptStructuredToolParamsSchema',
  'testStreamTools',
  'testStandardTextContentBlocks',
]) {
  test(`official integration: ${method}`, () => suite[method](), 900000)
}
