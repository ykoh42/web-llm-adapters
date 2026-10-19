import { z } from 'zod'
import { createDeepAgent } from 'deepagents'
import { InMemoryStore } from '@langchain/langgraph'
import { HumanMessage } from '@langchain/core/messages'
import { BrowserChatWebLLM, BrowserWebLLMEmbeddings } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../../shared/sampling.js'

test('real WebLLM embeddings rank LangGraph semantic memory', async ({ page }) => {
  const modelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId })
  const vector = await embeddings.embedQuery('A flower')
  expect(vector).toHaveLength(384)
  expect(vector.every(Number.isFinite)).toBe(true)
  const store = new InMemoryStore({ index: { embeddings, dims: 384 } })
  await store.put(['notes'], 'flower', { text: 'An orchid is a flowering plant.' })
  await store.put(['notes'], 'car', { text: 'A car is a vehicle with wheels and an engine.' })
  expect((await store.search(['notes'], { query: 'a flower', limit: 1 }))[0]?.key).toBe('flower')
})

test('real vision model accepts LangChain standard image blocks', async ({ page }) => {
  const modelId = 'Phi-3.5-vision-instruct-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const data = await page.evaluate(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 224
    canvas.height = 224
    const context = canvas.getContext('2d')!
    context.fillStyle = 'red'
    context.fillRect(0, 0, 224, 224)
    return canvas.toDataURL('image/png').split(',')[1]!
  })
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { max_tokens: 256, temperature: 0 },
  })
  const response = await model.invoke([
    new HumanMessage({
      content: [
        { type: 'text', text: 'What color is this image? Answer with the color.' },
        { type: 'image', source_type: 'base64', data, mime_type: 'image/png' },
      ],
    }),
  ])
  expect(response.text.toLowerCase()).toContain('red')
  const color = await model.withStructuredOutput(z.object({ color: z.string() })).invoke([
    new HumanMessage({
      content: [
        { type: 'text', text: 'Return the color of this image.' },
        { type: 'image', source_type: 'base64', data, mime_type: 'image/png' },
      ],
    }),
  ])
  expect(color.color.toLowerCase()).toContain('red')
})

test('real engine serializes concurrent inference, cancels generation and recovers', async ({
  page,
}) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const responses = await Promise.all([
    model.invoke('What is 1 + 1?'),
    model.invoke('What is 2 + 2?'),
  ])
  expect(responses[0]?.text).toContain('2')
  expect(responses[1]?.text).toContain('4')
  const cancel = new AbortController()
  const stream = await model.stream('Explain the history of mathematics in detail.', {
    signal: cancel.signal,
  })
  const first = await stream.next()
  expect(first.done).toBe(false)
  cancel.abort()
  await expect(stream.next()).rejects.toBeDefined()
  expect((await model.invoke('What is 3 + 3?')).text).toContain('6')
})

test('default Deep Agents subagents share the WebLLM engine and file state', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: { maxInputTokens: 32768 },
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const agent = createDeepAgent({ model })
  const tools: string[] = []
  const result = await agent.invoke(
    {
      messages: [
        {
          role: 'user',
          content:
            'Use the task tool with a general-purpose subagent to read /answer.txt. Then tell me the secret code from its report.',
        },
      ],
      files: {
        '/answer.txt': {
          content: ['The secret code is ORCHID-42.'],
          created_at: '2026-10-19',
          modified_at: '2026-10-19',
        },
      },
    },
    {
      callbacks: [
        {
          handleToolStart(...args) {
            if (args[6]) tools.push(args[6])
          },
        },
      ],
    },
  )
  expect(tools).toContain('task')
  expect(tools).toContain('read_file')
  expect(result.messages.at(-1)?.text).toContain('ORCHID-42')
})

test('the Deep Agents example writes and reads a file with native generation defaults', async ({
  page,
}) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: { maxInputTokens: 32768 },
    request: { extra_body: { enable_thinking: true } },
  })
  const agent = createDeepAgent({ model })
  const tools: string[] = []
  const result = await agent.invoke(
    {
      messages: [
        {
          role: 'user',
          content: 'Write a short greeting to /greeting.txt, then read it.',
        },
      ],
    },
    {
      callbacks: [
        {
          handleToolStart(...args) {
            if (args[6]) tools.push(args[6])
          },
        },
      ],
    },
  )
  expect(tools).toContain('write_file')
  expect(tools).toContain('read_file')
  expect(result.files?.['/greeting.txt']?.content.length).toBeGreaterThan(0)
  expect(result.messages.at(-1)?.type).toBe('ai')
  expect(result.messages.at(-1)?.text.length).toBeGreaterThan(0)
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId}: a WebLLM Worker runs default Deep Agents and handles cancellation`, async ({
    page,
  }) => {
    await page.evaluate((id) => window.checks.load(id, true), modelId)
    const model = new BrowserChatWebLLM(page, {
      modelId,
      profile: { maxInputTokens: 32768 },
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    })
    const agent = createDeepAgent({ model })
    const result = await agent.invoke({
      messages: [
        { role: 'user', content: 'Read /answer.txt and tell me the secret code exactly.' },
      ],
      files: {
        '/answer.txt': {
          content: ['The secret code is ORCHID-42.'],
          created_at: '2026-10-19',
          modified_at: '2026-10-19',
        },
      },
    })
    expect(result.messages.at(-1)?.text).toContain('ORCHID-42')
    const cancel = new AbortController()
    const stream = await model.stream('Explain the history of mathematics in detail.', {
      signal: cancel.signal,
    })
    expect((await stream.next()).done).toBe(false)
    cancel.abort()
    await expect(stream.next()).rejects.toBeDefined()
    expect((await model.invoke('What is 3 + 3?')).text).toContain('6')
  })
}
