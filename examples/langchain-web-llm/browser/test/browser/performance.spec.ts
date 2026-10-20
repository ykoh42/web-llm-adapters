import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from '@langchain/core/messages'
import type { BindToolsInput } from '@langchain/core/language_models/chat_models'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { sharedSchema } from '../../../../../test/support/schema.js'

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming
interface Workload {
  modelId: string
  name: string
  input: string | BaseMessage[]
  tools?: BindToolsInput[]
  minTokens?: number
}

const history = [
  new HumanMessage('Use lookup and report its answer briefly.'),
  new AIMessage({
    content: '',
    tool_calls: [
      {
        id: 'lookup',
        name: 'lookup',
        args: {
          samples: Array.from({ length: 128 }, (_, index) => index * 0.5),
          tiny: 1e-300,
          large: 1e300,
        },
      },
    ],
  }),
  new ToolMessage({ tool_call_id: 'lookup', content: 'The answer is 42.' }),
]

const workloads: Workload[] = [
  {
    modelId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    name: 'fast short text',
    input: 'What is 2 + 2? Answer briefly.',
  },
  {
    modelId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    name: 'fast long text',
    input:
      'Write a story about a sailor who finds a hidden island. Include an opening, a problem, and an ending.',
    minTokens: 64,
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    name: 'typed conditions',
    input: 'Use lookup to look up code 123.',
    tools: [
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: 'Look up a code.',
          parameters: {
            type: 'object',
            properties: { code: { type: 'string', oneOf: [{ pattern: '^1' }, { pattern: '^4' }] } },
            required: ['code'],
          },
        },
      },
    ],
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    name: 'shared schema references',
    input: 'Use lookup to look up code 123.',
    tools: [
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: 'Look up a code.',
          parameters: sharedSchema(),
        },
      },
    ],
  },
  { modelId: 'Qwen3.5-4B-q4f16_1-MLC', name: 'text', input: 'What is 2 + 2? Answer briefly.' },
  { modelId: 'Qwen3.5-4B-q4f16_1-MLC', name: 'JSON tool history', input: history },
  {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    name: 'JSON tool history',
    input: history,
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    name: 'schema union',
    input: 'Use lookup to look up code 123.',
    tools: [
      {
        type: 'function',
        function: {
          name: 'lookup',
          description: 'Look up a code.',
          parameters: {
            type: 'object',
            oneOf: [
              { properties: { code: { const: '123' } }, required: ['code'] },
              { properties: { code: { const: '456' } }, required: ['code'] },
            ],
          },
        },
      },
    ],
  },
]

for (const { modelId, name, input, tools, minTokens } of workloads) {
  test(`${modelId}: browser transport stays close to native WebLLM on the same workload: ${name}`, async ({
    page,
  }, testInfo) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(() => {
      const engine = window.checks.engine()
      Object.assign(window, { nativeRequests: [] })
      for (const endpoint of [engine.chat.completions, engine.completions]) {
        Object.assign(endpoint, {
          create: new Proxy(endpoint.create, {
            apply(target, self, args) {
              ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
                structuredClone(args[0]),
              )
              return Reflect.apply(target, self, args)
            },
          }),
        })
      }
    })
    const raw: WebLLMChunk[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => raw.push(chunk),
    })
    const llm = tools ? model.bindTools(tools) : model
    await llm.invoke(input)
    const request = await page.evaluate(
      () => (window as unknown as { nativeRequests: NativeRequest[] }).nativeRequests[0]!,
    )
    const results: { native: number; adapter: number; tokens: number }[] = []
    for (let index = 0; index < 3; index++) {
      const native = await page.evaluate(async (request) => {
        const engine = window.checks.engine()
        await engine.resetChat()
        const start = performance.now()
        let text = ''
        let tokens = 0
        const stream =
          'messages' in request
            ? await engine.chat.completions.create(request)
            : await engine.completions.create(request)
        for await (const chunk of stream) {
          if (chunk.usage) tokens = chunk.usage.completion_tokens
          const choice = chunk.choices[0]
          if (choice) text += 'delta' in choice ? (choice.delta.content ?? '') : choice.text
        }
        return { elapsed: performance.now() - start, text, tokens }
      }, request)
      await page.evaluate(() => window.checks.engine().resetChat())
      raw.length = 0
      const start = performance.now()
      const response = await llm.invoke(input)
      const elapsed = performance.now() - start
      if (tools)
        expect(response.tool_calls).toMatchObject([{ name: 'lookup', args: { code: '123' } }])
      const adapted = raw
        .map((chunk) => {
          const choice = chunk.choices[0]
          if (!choice) return ''
          return 'delta' in choice ? (choice.delta.content ?? '') : choice.text
        })
        .join('')
      expect(adapted).toBe(native.text)
      expect(response.usage_metadata?.output_tokens).toBe(native.tokens)
      if (minTokens !== undefined) expect(native.tokens).toBeGreaterThanOrEqual(minTokens)
      results.push({ native: native.elapsed, adapter: elapsed, tokens: native.tokens })
    }
    const median = (values: number[]) => values.sort((a, b) => a - b)[1]!
    const baseline = median(results.map((result) => result.native))
    const adapted = median(results.map((result) => result.adapter))
    const path = testInfo.outputPath('performance.json')
    await writeFile(
      path,
      JSON.stringify({ modelId, workload: name, results, ratio: adapted / baseline }, null, 2),
    )
    await testInfo.attach('performance', { path, contentType: 'application/json' })
    expect(adapted).toBeLessThanOrEqual(baseline * 1.1 + 50)
  })
}
