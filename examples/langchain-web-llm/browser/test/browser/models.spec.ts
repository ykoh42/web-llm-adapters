import { modelCheck, recordThinking } from '../../../../../scripts/release/support.js'
import { sampling } from '../../../sampling.js'
import { createDeepAgent } from 'deepagents'
import { createAgent } from 'langchain'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import type {} from '../../src/main.js'
import { writeFile } from 'node:fs/promises'
import { hasToolResult, nativeOutput } from '../../../../../test/support/native-output.js'

const gemmaModel = 'gemma-4-E2B-it-q4f16_1-MLC'
const models = [
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  gemmaModel,
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

const checks = selected.flatMap((item) =>
  (item.modelId === 'Qwen2.5-7B-Instruct-q4f16_1-MLC' ? [false, true] : [false]).map(
    (constrainedToolCalls) => ({ ...item, constrainedToolCalls }),
  ),
)

for (const { modelId, prompt, constrainedToolCalls } of checks) {
  const protocol = modelId === 'Qwen2.5-7B-Instruct-q4f16_1-MLC' && !constrainedToolCalls
  const title = protocol
    ? 'default Deep Agents preserves native tool output'
    : 'Deep Agents tools and final answer'
  const feature = constrainedToolCalls ? 'constrained-tools' : 'tools'
  test(
    `${modelId}: ${title}` + (constrainedToolCalls ? ' with constraints' : ''),
    modelCheck(modelId, protocol ? 'tool-protocol' : feature),
    async ({ page }, testInfo) => {
      await page.evaluate((id) => window.checks.load(id), modelId)
      const profile = await page.evaluate(() => window.checks.model().profile)
      const configuredContext = await page.evaluate(() => window.checks.contextWindowSize())
      expect(profile.maxInputTokens).toBe(configuredContext)
      await page.evaluate(() => {
        const engine = window.checks.engine()
        const create = engine.chat.completions.create.bind(engine.chat.completions)
        const complete = engine.completions.create.bind(engine.completions)
        Object.assign(window, { nativeRequests: [], nativePrompts: [] })
        engine.chat.completions.create = new Proxy(create, {
          apply(target, self, args) {
            ;(window as unknown as { nativeRequests: unknown[] }).nativeRequests.push(
              structuredClone(args[0]),
            )
            return Reflect.apply(target, self, args)
          },
        })
        engine.completions.create = new Proxy(complete, {
          apply(target, self, args) {
            ;(window as unknown as { nativePrompts: unknown[] }).nativePrompts.push(
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
        thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
        profile,
        onChunk: (chunk) => native.push(chunk),
        request: {
          ...sampling(modelId, thinking),
          max_tokens: 4096,
          seed: 7,
          constrainedToolCalls,
        },
      })
      const agent = createDeepAgent({ model })
      let toolCalls = 0
      let outcome: { result: Awaited<ReturnType<typeof agent.invoke>> } | { error: unknown }
      try {
        outcome = await agent
          .invoke(
            {
              messages: [{ role: 'user', content: prompt }],
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
                  handleToolStart() {
                    toolCalls++
                  },
                },
              ],
            },
          )
          .then(
            (result) => ({ result }),
            (error: unknown) => ({ error }),
          )
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
      const requests = await page.evaluate(
        () =>
          (window as unknown as { nativeRequests: ChatCompletionRequestStreaming[] })
            .nativeRequests,
      )
      if ('error' in outcome) {
        if (!protocol) throw outcome.error
        expect(outcome.error).toMatchObject({ message: 'Qwen returned invalid tool JSON.' })
        expect(toolCalls).toBe(0)
        expect(requests).toHaveLength(1)
        expect(requests[0]).not.toHaveProperty('response_format')
        const raw = native
          .flatMap((chunk) =>
            chunk.choices.map((choice) =>
              'delta' in choice ? (choice.delta.content ?? '') : choice.text,
            ),
          )
          .join('')
        const direct = await page.evaluate(async (request) => {
          const engine = window.checks.engine()
          await engine.resetChat()
          let text = ''
          for await (const chunk of await engine.chat.completions.create(request))
            text += chunk.choices[0]?.delta.content ?? ''
          return text
        }, requests[0]!)
        await testInfo.attach('native-default', {
          body: JSON.stringify({ raw, direct, error: String(outcome.error) }),
          contentType: 'application/json',
        })
        expect(direct.startsWith(raw)).toBe(true)
        expect(raw.match(/<tool_call>/g)).toHaveLength(1)
        expect(raw.match(/<\/tool_call>/g)).toHaveLength(1)
        const source = raw.match(/<tool_call>([\s\S]*)<\/tool_call>/)?.[1]
        expect(source).toBeDefined()
        expect(() => JSON.parse(source!)).toThrow(SyntaxError)
        return
      }
      const nativeRequests = await page.evaluate(() => {
        const records = window as unknown as {
          nativeRequests: ChatCompletionRequestStreaming[]
          nativePrompts: CompletionCreateParamsStreaming[]
        }
        return records.nativeRequests.length ? records.nativeRequests : records.nativePrompts
      })
      const firstRequest = nativeRequests[0]
      if (constrainedToolCalls)
        expect(firstRequest).toHaveProperty('response_format.type', 'structural_tag')
      else expect(firstRequest).not.toHaveProperty('response_format')
      const output = outcome.result
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
      expect(hasToolResult(nativeRequests.slice(1), 'ORCHID-42')).toBe(true)
      expect(result.at(-1)?.type).toBe('ai')
      const answer = result.at(-1)!
      const thoughts = result
        .filter((message) => message.type === 'ai')
        .map((message) =>
          Array.isArray(message.content)
            ? message.content
                .flatMap((block) =>
                  block.type === 'reasoning' && typeof block.reasoning === 'string'
                    ? [block.reasoning]
                    : [],
                )
                .join('')
            : '',
        )
      const prefix = modelId.startsWith('Qwen3.5-')
        ? '<think>'
        : nativeRequests.map((request) =>
            'prompt' in request && request.prompt.endsWith('<|channel>thought\n')
              ? '<|channel>thought\n'
              : '',
          )
      expect(answer.text.trim()).not.toBe('')
      expect({ text: answer.text, thinking: thoughts }).toEqual(
        nativeOutput(native, modelId, prefix),
      )
      recordThinking(testInfo, modelId, thoughts, nativeRequests)
      const finishes = native
        .flatMap((chunk) => chunk.choices.map((choice) => choice.finish_reason))
        .filter(Boolean)
      expect(finishes.at(-1)).toBe('stop')
      expect(finishes).not.toContain('length')
      if (modelId === gemmaModel) {
        const reasoning = 'Prior lookup reasoning should survive.'
        const model = new BrowserChatWebLLM(page, {
          modelId,
          profile,
          request: {
            ...sampling(modelId, true),
            max_tokens: 1024,
            seed: 7,
            extra_body: { enable_thinking: true, preserve_thinking: true },
          },
        })
        const history = [
          new HumanMessage('Look up A7.'),
          new AIMessage({
            content: [{ type: 'reasoning', reasoning }],
            tool_calls: [
              { id: 'lookup-1', name: 'lookup', args: { code: 'A7' }, type: 'tool_call' },
            ],
          }),
          new ToolMessage({ tool_call_id: 'lookup-1', name: 'lookup', content: 'ORCHID-42' }),
          new HumanMessage('What was the lookup result? Reply with only the value.'),
        ]
        const answer = await model.invoke(history, {
          request: { extra_body: { enable_thinking: true, enable_latency_breakdown: true } },
        })
        const request = await page.evaluate(() =>
          (window as unknown as { nativePrompts: unknown[] }).nativePrompts.at(-1),
        )

        expect(request).toMatchObject({
          prompt: expect.stringContaining('<|think|>\n'),
          extra_body: { enable_latency_breakdown: true },
        })
        expect(request).toMatchObject({ prompt: expect.stringContaining(reasoning) })
        expect(request).not.toHaveProperty('extra_body.enable_thinking')
        expect(request).not.toHaveProperty('extra_body.preserve_thinking')
        expect(answer.text).toContain('ORCHID-42')
      }
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
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
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
    thinkingPrefilled: true,
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
  const [, thinking, text] = /^([\s\S]*?)<\/think>([\s\S]*)$/.exec(raw) ?? []
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
