import { Agent, TextBlock } from '@strands-agents/sdk'
import { test, expect } from '../../../test/browser.js'
import { BrowserModel, pcmAudio, type WebLLMChunk } from 'strands-webllm'

test('Gemma returns constrained JSON, native logprobs and PCM results through BrowserModel', async ({
  page,
}) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    const chunks: WebLLMChunk[] = []
    const model = new BrowserModel(page, {
      modelId,
      maxTokens: 128,
      temperature: 0,
      onChunk: (chunk) => chunks.push(chunk),
      request: {
        logprobs: true,
        top_logprobs: 2,
        response_format: {
          type: 'json_object',
          schema: JSON.stringify({
            type: 'object',
            properties: { word: { const: 'hello' } },
            required: ['word'],
          }),
        },
      },
    })
    const jsonRequest = model.getConfig().request!
    model.updateConfig({
      request: { response_format: { type: 'text' }, extra_body: { enable_thinking: true } },
      maxTokens: 512,
    })
    const agent = new Agent({ model, printer: false })
    const thought = await agent.invoke('What is 2 plus 2?')
    expect(thought.lastMessage.content.some((b) => b.type === 'reasoningBlock' && b.text)).toBe(
      true,
    )
    model.updateConfig({ request: jsonRequest, maxTokens: 128 })
    chunks.length = 0
    const result = await agent.invoke('Return JSON with word hello.')
    const text = result.lastMessage.content
      .filter((b) => b.type === 'textBlock')
      .map((b) => b.text)
      .join('')
    expect(JSON.parse(text)).toEqual({ word: 'hello' })
    expect(chunks.every((chunk) => chunk.object === 'chat.completion.chunk')).toBe(true)
    expect(
      chunks.some((chunk) => chunk.choices.some((choice) => choice.logprobs?.content?.length)),
    ).toBe(true)
    expect(chunks.at(-1)?.usage?.total_tokens).toBeGreaterThan(0)

    model.updateConfig({ request: {} })
    chunks.length = 0
    const audio = pcmAudio(new Float32Array(16000), 16000)
    const heard = await new Agent({ model, printer: false }).invoke([
      new TextBlock('Describe any sound you hear in one short sentence.'),
      audio,
    ])
    expect(heard.stopReason).toBe('endTurn')
    expect(heard.lastMessage.content.some((b) => b.type === 'textBlock' && b.text.length)).toBe(
      true,
    )
    expect(chunks.at(-1)?.usage?.prompt_tokens).toBeGreaterThan(0)
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})
