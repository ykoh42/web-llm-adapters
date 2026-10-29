import { writeFile } from 'node:fs/promises'
import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { test, expect } from './browser.js'
import type {} from './checks.js'

for (const modelId of [
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'Qwen3-0.6B-q4f16_1-MLC',
]) {
  test(`${modelId}: preserves native tool selection and arguments`, async ({ page }, info) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate((id) => window.piChecks.load(id), modelId)
    const evidence = await page.evaluate(async (modelId) => {
      const agent = await window.piChecks.createAgent(true)
      const { models, model } = await window.piChecks.provider(true)
      const systemPrompt =
        modelId === 'Qwen3.5-2B-q4f16_1-MLC'
          ? `<instructions>\n${agent.state.systemPrompt}\n</instructions>`
          : agent.state.systemPrompt
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
      return { request: payload, native, message, systemPrompt }
    }, modelId)
    await writeFile(info.outputPath('tool-choice.json'), JSON.stringify(evidence, null, 2))
    const { request, native, message, systemPrompt } = evidence
    expect(request.extra_body?.enable_thinking).toBe(true)
    const system = request.messages[0]?.content
    expect(request.messages[0]?.role).toBe('system')
    expect(typeof system).toBe('string')
    if (modelId.startsWith('Qwen3.5-'))
      expect(String(system).endsWith(`\n\n${systemPrompt}`)).toBe(true)
    else expect(String(system).startsWith(`${systemPrompt}\n\n`)).toBe(true)
    const declaration = /<tools>\n([^\n]+)\n<\/tools>/.exec(String(system))?.[1]
    expect(JSON.parse(declaration ?? 'null')).toMatchObject({
      type: 'function',
      function: {
        name: 'lookup',
        parameters: {
          type: 'object',
          required: ['code'],
          properties: { code: { type: 'string' } },
        },
      },
    })
    expect(request.messages.at(-1)).toEqual({
      role: 'user',
      content: 'Look up A7 and tell me the answer.',
    })
    const calls = message.content.filter((part) => part.type === 'toolCall')
    const expectedCalls = [...native.matchAll(/<tool_call>\n([\s\S]*?)\n<\/tool_call>/g)].map(
      (match) => {
        if (!modelId.startsWith('Qwen3.5-')) return JSON.parse(match[1]!)
        const call =
          /^<function=([^>]+)>\n<parameter=code>\n([\s\S]*?)\n<\/parameter>\n<\/function>$/.exec(
            match[1]!,
          )
        expect(call).not.toBeNull()
        return { name: call![1], arguments: { code: call![2] } }
      },
    )
    expect(calls.map(({ name, arguments: args }) => ({ name, arguments: args }))).toEqual(
      expectedCalls,
    )
    expect(message.stopReason).toBe(expectedCalls.length ? 'toolUse' : 'stop')
    const output = modelId.startsWith('Qwen3.5-') ? '<think>' + native : native
    const match = /^<think>([\s\S]*?)<\/think>([\s\S]*)$/.exec(output)
    const thinking = match?.[1] ?? ''
    const text = (match?.[2] ?? native).replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
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
}
