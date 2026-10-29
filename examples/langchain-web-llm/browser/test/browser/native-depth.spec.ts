import { writeFile } from 'node:fs/promises'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'

type Tree = number | { next: Tree }

declare global {
  interface Window {
    deepNativeRequests: ChatCompletionRequestStreaming[]
  }
}

test('Hermes keeps deep tool history identical to a native WebLLM request', async ({
  page,
}, info) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  let tree: Tree = 42
  for (let level = 0; level < 3000; level++) tree = { next: tree }
  const input = { code: 'A7', tree }
  const raw = JSON.stringify([{ name: 'lookup', arguments: { tree, code: 'A7' } }])
  const request: ChatCompletionRequestStreaming = {
    model: modelId,
    messages: [
      { role: 'user', content: 'Use lookup.' },
      {
        role: 'assistant',
        content: raw,
        tool_calls: [
          {
            id: 'call',
            type: 'function',
            function: { name: 'lookup', arguments: JSON.stringify(input) },
          },
        ],
      },
      { role: 'tool', tool_call_id: 'call', content: 'Done.' },
      { role: 'user', content: 'Reply only OK.' },
    ],
    temperature: 0,
    max_tokens: 4096,
    seed: 7,
    stream: true,
    stream_options: { include_usage: true },
    n: 1,
  }
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(
    (modelId) =>
      window.webllmLangChainBridge!.withEngine(async (engine) => {
        await engine.reload(modelId, { context_window_size: 32768 })
        window.deepNativeRequests = []
        const endpoint = engine.chat.completions
        endpoint.create = new Proxy(endpoint.create, {
          apply(target, self, args) {
            window.deepNativeRequests.push(structuredClone(args[0]))
            return Reflect.apply(target, self, args)
          },
        })
      }),
    modelId,
  )
  let observed = ''
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { temperature: 0, max_tokens: 4096, seed: 7 },
    onChunk(chunk) {
      const choice = chunk.choices[0]
      if (choice && 'delta' in choice) observed += choice.delta.content ?? ''
    },
  })
  const result = await model.invoke([
    new HumanMessage('Use lookup.'),
    new AIMessage({ content: raw, tool_calls: [{ id: 'call', name: 'lookup', args: input }] }),
    new ToolMessage({ tool_call_id: 'call', content: 'Done.' }),
    new HumanMessage('Reply only OK.'),
  ])
  const [prepared] = await page.evaluate(() => window.deepNativeRequests)
  const evidence = info.outputPath('native-depth.json')
  await writeFile(
    evidence,
    JSON.stringify({ prepared, expected: request, observed, text: result.text }, null, 2),
  )
  await info.attach('native-depth', { path: evidence, contentType: 'application/json' })
  expect(prepared?.messages[1]?.content === raw).toBe(true)
  expect(prepared).toEqual(request)
  expect(observed).toContain('OK')
  expect(result.text).toBe(observed)
  expect(result.response_metadata.finish_reason).toBe('stop')
})
