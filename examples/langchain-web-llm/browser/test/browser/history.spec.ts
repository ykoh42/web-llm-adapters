import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

declare global {
  interface Window {
    historyRequests: ChatCompletionRequestStreaming[]
  }
}

test('Qwen3.5 rejects ambiguous history before inference and keeps valid JSON keys', async ({
  page,
}, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id, true), modelId)
  await page.evaluate(() => {
    window.historyRequests = []
    const endpoint = window.checks.engine().chat.completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.historyRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
  const native: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => native.push(chunk),
  })
  const user = new HumanMessage('Read the value.')
  const result = new ToolMessage({ content: '42', tool_call_id: 'call' })
  const end = new HumanMessage('What is 17 * 23?')
  const invalid = new AIMessage({
    content: '',
    tool_calls: [{ id: 'call', name: 'lookup', args: { 'a>b': 42 } }],
  })
  await expect(model.invoke([user, invalid, result, end])).rejects.toThrow('parameter names')
  expect(await page.evaluate(() => window.historyRequests)).toHaveLength(0)
  const input = { payload: { 'a>b': 42 }, text: 'a > b' }
  const before = structuredClone(input)
  const valid = new AIMessage({
    content: '',
    tool_calls: [{ id: 'call', name: 'lookup', args: input }],
  })
  const response = await model.invoke([user, valid, result, end])
  const requests = await page.evaluate(() => window.historyRequests)
  await info.attach('history-recovery', {
    body: JSON.stringify({
      invalid: invalid.toDict(),
      valid: valid.toDict(),
      requests,
      response,
      native,
    }),
    contentType: 'application/json',
  })
  expect(response.text).toContain('391')
  const raw = native
    .flatMap((chunk) =>
      chunk.choices.map((choice) =>
        'delta' in choice ? (choice.delta.content ?? '') : choice.text,
      ),
    )
    .join('')
  const thought = /<think>([\s\S]*?)<\/think>/.exec(raw)?.[1]
  expect(thought).toBeDefined()
  expect(
    response.contentBlocks
      .filter((block) => block.type === 'reasoning')
      .map((block) => block.reasoning)
      .join(''),
  ).toBe(thought)
  expect(requests).toHaveLength(1)
  expect(requests[0]?.extra_body?.enable_thinking).toBe(true)
  const replay = requests[0]?.messages.find((message) => message.role === 'assistant')?.content
  expect(replay).toContain('"a>b": 42')
  expect(replay).toContain('a > b')
  expect(response.usage_metadata?.output_tokens).toBeGreaterThan(0)
  expect(valid.tool_calls?.[0]?.args).toEqual(before)
  expect(input).toEqual(before)
})
