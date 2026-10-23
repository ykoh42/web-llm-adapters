import { HumanMessage } from '@langchain/core/messages'
import { scrubUnsupportedMultimodalContent } from 'deepagents'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { createAgent } from 'langchain'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

test('Hermes image limits let Deep Agents recover without changing stored messages', async ({
  page,
}) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const url = await page.evaluate(() => {
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 2
    return canvas.toDataURL('image/png')
  })
  await page.evaluate(() => {
    const endpoint = window.checks.engine().chat.completions
    Object.assign(window, { hermesImageRequests: [] })
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.hermesImageRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { max_tokens: 4096, temperature: 0, seed: 7 },
  })
  const content = [
    { type: 'text', text: 'What is 6 * 7?' },
    { type: 'image', url },
  ]
  const original = new HumanMessage({ content })
  await expect(model.invoke([original])).rejects.toThrow(
    'user message only supports string content',
  )
  const messages = scrubUnsupportedMultimodalContent([original], model)
  expect(original.content).toEqual(content)
  expect(messages[0]?.contentBlocks.every((block) => block.type === 'text')).toBe(true)
  const answer = await model.invoke(messages)
  expect(answer.text).toContain('42')
  const requests = await page.evaluate(() => window.hermesImageRequests)
  expect(requests).toHaveLength(2)
  expect(requests[1]?.messages).toEqual([{ role: 'user', content: messages[0]!.text }])
  const native = await page.evaluate(async (request) => {
    const engine = window.checks.engine()
    await engine.resetChat()
    let text = ''
    let usage
    let reason
    for await (const chunk of await engine.chat.completions.create(request)) {
      text += chunk.choices[0]?.delta.content ?? ''
      reason = chunk.choices[0]?.finish_reason ?? reason
      usage = chunk.usage ?? usage
    }
    return { text, usage, reason }
  }, requests[1]!)
  expect(native.reason).toBe('stop')
  expect(answer.text).toBe(native.text)
  expect(answer.usage_metadata?.input_tokens).toBe(native.usage?.prompt_tokens)
  expect(answer.usage_metadata?.output_tokens).toBe(native.usage?.completion_tokens)
})

test('Gemma thinking selects a supported agent response strategy', async ({ page }, info) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    window.gemmaStructuredRequests = []
    const endpoint = window.checks.engine().completions
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.gemmaStructuredRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
  let raw: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => raw.push(chunk),
  })
  expect(model.profile).toMatchObject({ structuredOutput: false, reasoningOutput: true })
  const answer = await model.invoke('What is 17 * 23? Reply briefly.')
  expect(answer.text).toContain('391')
  expect(answer.contentBlocks).toContainEqual(
    expect.objectContaining({ type: 'reasoning', reasoning: expect.stringMatching(/\S/) }),
  )
  const thinking = raw
  raw = []
  const agent = createAgent({ model, responseFormat: z.object({ answer: z.number() }) })
  const result = await agent.invoke({
    messages: [{ role: 'user', content: 'Return answer set to six times seven.' }],
  })
  expect(result.structuredResponse).toEqual({ answer: 42 })
  const requests = await page.evaluate(() => window.gemmaStructuredRequests)
  expect(requests).toHaveLength(2)
  for (const request of requests) {
    expect(request).toMatchObject({ model: modelId, max_tokens: 4096, seed: 7 })
    expect(request.prompt).toContain('<|think|>')
    expect(request).not.toHaveProperty('response_format')
  }
  expect(requests[1]?.prompt).toMatch(/<\|tool_call>call:[^\s{]+\{$/)
  expect(raw.length).toBeGreaterThan(0)
  await info.attach('native-structured-strategy', {
    body: JSON.stringify(
      { requests, thinking, structured: raw, result: result.structuredResponse },
      null,
      2,
    ),
    contentType: 'application/json',
  })
})

declare global {
  interface Window {
    hermesImageRequests: ChatCompletionRequestStreaming[]
    gemmaStructuredRequests: CompletionCreateParamsStreaming[]
  }
}
