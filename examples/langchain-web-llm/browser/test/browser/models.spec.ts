import { modelCheck } from '../../../../../scripts/release/support.js'
import { sampling } from '../../../sampling.js'
import { createDeepAgent } from 'deepagents'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import type {} from '../../src/main.js'
import { writeFile } from 'node:fs/promises'

const models = [
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
]

const defaultPrompt = 'Read /answer.txt and tell me the secret code exactly.'
const cases = [
  ...models.map((modelId) => ({
    modelId,
    prompt: defaultPrompt,
  })),
  ...['Qwen3.5-2B-q4f16_1-MLC', 'Qwen3.5-0.8B-q4f16_1-MLC', 'Qwen3-1.7B-q4f16_1-MLC'].map(
    (modelId) => ({
      modelId,
      prompt: 'Use read_file to read /answer.txt, then report the secret code from its result.',
    }),
  ),
]
const selected =
  process.env.MODEL_IDS?.split(',').map(
    (modelId) =>
      cases.find((item) => item.modelId === modelId) ?? { modelId, prompt: defaultPrompt },
  ) ?? cases

for (const { modelId, prompt } of selected) {
  test(
    `${modelId}: default Deep Agents tools and final answer`,
    modelCheck(
      modelId,
      'tools',
      ...(!modelId.startsWith('Qwen2.5-') ? (['thinking'] as const) : []),
    ),
    async ({ page }, testInfo) => {
      await page.evaluate((id) => window.checks.load(id), modelId)
      const profile = await page.evaluate(() => window.checks.model().profile)
      const configuredContext = await page.evaluate(() => window.checks.contextWindowSize())
      expect(profile.maxInputTokens).toBe(configuredContext)
      await page.evaluate(() => {
        const engine = window.checks.engine()
        const create = engine.chat.completions.create.bind(engine.chat.completions)
        Object.assign(window, { nativeRequests: [] })
        engine.chat.completions.create = new Proxy(create, {
          apply(target, self, args) {
            ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
              structuredClone(args[0]),
            )
            return Reflect.apply(target, self, args)
          },
        })
      })
      const thinking = !modelId.startsWith('Qwen2.5-')
      const native: WebLLMChunk[] = []
      const model = new BrowserChatWebLLM(page, {
        modelId,
        profile,
        onChunk: (chunk) => native.push(chunk),
        request: { ...sampling(modelId, thinking), max_tokens: 4096, seed: 7 },
      })
      const agent = createDeepAgent({ model })
      let output: Awaited<ReturnType<typeof agent.invoke>>
      try {
        output = await agent.invoke({
          messages: [{ role: 'user', content: prompt }],
          files: {
            '/answer.txt': {
              content: ['The secret code is ORCHID-42.'],
              created_at: '2026-10-19',
              modified_at: '2026-10-19',
            },
          },
        })
      } finally {
        await writeFile(
          testInfo.outputPath('requests.json'),
          JSON.stringify(
            await page.evaluate(
              () => (window as unknown as { nativeRequests: unknown[] }).nativeRequests,
            ),
            null,
            2,
          ),
        )
        const path = testInfo.outputPath('native.json')
        await writeFile(path, JSON.stringify(native, null, 2))
        await testInfo.attach('native', {
          path,
          contentType: 'application/json',
        })
      }
      const result = output.messages.map((message) => ({
        type: message.type,
        text: message.text,
        content: message.content,
        ...('tool_calls' in message && { tool_calls: message.tool_calls }),
      }))
      await writeFile(testInfo.outputPath('messages.json'), JSON.stringify(result, null, 2))
      await testInfo.attach('messages', {
        body: JSON.stringify(result, null, 2),
        contentType: 'application/json',
      })
      expect(
        result.some((message) => message.type === 'tool' && message.text.includes('ORCHID-42')),
      ).toBe(true)
      expect(result.at(-1)?.type).toBe('ai')
      expect(result.at(-1)?.text).toContain('ORCHID-42')
      const finishes = native
        .flatMap((chunk) => chunk.choices.map((choice) => choice.finish_reason))
        .filter(Boolean)
      expect(finishes.at(-1)).toBe('stop')
      expect(finishes).not.toContain('length')
      if (!modelId.startsWith('Qwen2.5-'))
        expect(
          result.some(
            (message) =>
              Array.isArray(message.content) &&
              message.content.some(
                (block) =>
                  block.type === 'reasoning' &&
                  typeof block.reasoning === 'string' &&
                  block.reasoning.length > 0,
              ),
          ),
        ).toBe(true)
    },
  )
}

for (const modelId of ['Qwen3.5-2B-q4f16_1-MLC', 'Qwen3.5-0.8B-q4f16_1-MLC']) {
  test(`${modelId}: LangChain Agent tools, thinking and final answer`, async ({ page }, info) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const native: WebLLMChunk[] = []
    const inputs: { code: string }[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => native.push(chunk),
    })
    const lookup = tool(
      ({ code }) => {
        inputs.push({ code })
        return `The answer for ${code} is ORCHID-42.`
      },
      {
        name: 'lookup',
        description: 'Look up the answer for a code.',
        schema: z.object({ code: z.string() }),
      },
    )
    const agent = createAgent({
      model,
      tools: [lookup],
      systemPrompt: 'Use lookup when asked to look up a code. Report its returned answer.',
    })
    let output: Awaited<ReturnType<typeof agent.invoke>>
    try {
      output = await agent.invoke({
        messages: [{ role: 'user', content: 'Look up A7 and tell me the answer.' }],
      })
    } finally {
      await writeFile(info.outputPath('native.json'), JSON.stringify({ native, inputs }, null, 2))
    }
    const messages = output.messages.map((message) => ({
      type: message.type,
      text: message.text,
      content: message.content,
    }))
    await writeFile(info.outputPath('messages.json'), JSON.stringify(messages, null, 2))
    expect(inputs).toEqual([{ code: 'A7' }])
    expect(
      messages.some((message) => message.type === 'tool' && message.text.includes('ORCHID-42')),
    ).toBe(true)
    expect(messages.at(-1)?.type).toBe('ai')
    expect(messages.at(-1)?.text).toContain('ORCHID-42')
    expect(
      messages.some(
        (message) =>
          Array.isArray(message.content) &&
          message.content.some(
            (block) =>
              block.type === 'reasoning' &&
              typeof block.reasoning === 'string' &&
              block.reasoning.length > 0,
          ),
      ),
    ).toBe(true)
    const finishes = native
      .flatMap((chunk) => chunk.choices.map((choice) => choice.finish_reason))
      .filter(Boolean)
    expect(finishes.at(-1)).toBe('stop')
    expect(finishes).not.toContain('length')
  })
}

test('Qwen3.5-2B: thinking and answer match native WebLLM output', async ({ page }, info) => {
  const modelId = 'Qwen3.5-2B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const native: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => native.push(chunk),
  })
  const result = await model.invoke('What is 17 * 23 + 19 * 29?')
  const raw = native
    .flatMap((chunk) =>
      chunk.choices.map((choice) =>
        'delta' in choice ? (choice.delta.content ?? '') : choice.text,
      ),
    )
    .join('')
  await writeFile(info.outputPath('thinking.json'), JSON.stringify({ result, raw }, null, 2))
  const [, thinking, text] = /^<think>([\s\S]*?)<\/think>([\s\S]*)$/.exec(raw) ?? []
  expect(thinking).toBeDefined()
  expect(result.text).toContain('942')
  expect(result.text).toBe(text)
  const reasoning = Array.isArray(result.content)
    ? result.content
        .flatMap((block) =>
          block.type === 'reasoning' && typeof block.reasoning === 'string'
            ? [block.reasoning]
            : [],
        )
        .join('')
    : undefined
  expect(reasoning).toBe(thinking)
})
