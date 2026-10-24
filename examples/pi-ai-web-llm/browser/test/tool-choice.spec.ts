import { writeFile } from 'node:fs/promises'
import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { test, expect } from './browser.js'
import type {} from './checks.js'

test('Qwen3.5-2B preserves a native text answer when tools are available', async ({
  page,
}, info) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-2B-q4f16_1-MLC'))
  const evidence = await page.evaluate(async () => {
    const agent = await window.piChecks.createAgent(true)
    const { models, model, engine } = await window.piChecks.provider(true)
    const systemPrompt = `<instructions>\n${agent.state.systemPrompt}\n</instructions>`
    let payload: ChatCompletionRequestStreaming | undefined
    let native = ''
    const message = await models.completeSimple(
      model,
      {
        systemPrompt,
        tools: agent.state.tools,
        messages: [{ role: 'user', content: 'Look up A7 and tell me the answer.', timestamp: 0 }],
      },
      {
        reasoning: 'low',
        onPayload: (request) => {
          payload = structuredClone(request) as ChatCompletionRequestStreaming
        },
        onProviderStreamEvent: (event) => {
          const chunk = event as ChatCompletionChunk
          native += chunk.choices[0]?.delta.content ?? ''
        },
      },
    )
    if (!payload) throw new Error('Missing native request.')
    const request = payload
    const direct = await window.piChecks.withEngine(engine, async (engine) => {
      await engine.resetChat()
      let raw = ''
      for await (const chunk of await engine.chat.completions.create(request))
        raw += chunk.choices[0]?.delta.content ?? ''
      return raw
    })
    return { request, native, direct, message, systemPrompt }
  })
  await writeFile(info.outputPath('tool-choice.json'), JSON.stringify(evidence, null, 2))
  const { request, native, direct, message, systemPrompt } = evidence
  expect(request.extra_body?.enable_thinking).toBe(true)
  const system = request.messages[0]?.content
  expect(request.messages[0]?.role).toBe('system')
  expect(typeof system).toBe('string')
  expect(String(system).endsWith(`\n\n${systemPrompt}`)).toBe(true)
  const declaration = /<tools>\n([^\n]+)\n<\/tools>/.exec(String(system))?.[1]
  expect(JSON.parse(declaration ?? 'null')).toMatchObject({
    type: 'function',
    function: {
      name: 'lookup',
      parameters: { type: 'object', required: ['code'], properties: { code: { type: 'string' } } },
    },
  })
  expect(request.messages.at(-1)).toEqual({
    role: 'user',
    content: 'Look up A7 and tell me the answer.',
  })
  expect(native).toBe(direct)
  expect(native).not.toContain('<tool_call>')
  expect(message.stopReason).toBe('stop')
  expect(message.content.some((part) => part.type === 'toolCall')).toBe(false)
  const match = /^<think>([\s\S]*?)<\/think>([\s\S]*)$/.exec(native)
  const thinking = match?.[1] ?? ''
  const text = match?.[2] ?? native
  expect(
    message.content
      .filter((part) => part.type === 'thinking')
      .map((part) => part.thinking)
      .join(''),
  ).toBe(thinking)
  expect(
    message.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join(''),
  ).toBe(text)
})
