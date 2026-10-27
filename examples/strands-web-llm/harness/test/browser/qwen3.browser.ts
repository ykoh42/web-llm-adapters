import { modelCheck } from '../../../../../scripts/release/support.js'
import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { BeforeModelCallEvent, FunctionTool, type Message } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { Page } from '@playwright/test'
import { recordNative, replayNative } from '../../fixtures/native.js'
import { sampling } from '../../../sampling.js'
import { nativeOutput } from '../../../../../test/support/native-output.js'

const cases = [
  ...[
    'Qwen3-1.7B-q4f16_1-MLC',
    'Qwen3-0.6B-q4f16_1-MLC',
    'Qwen3-4B-q4f16_1-MLC',
    'Qwen3-8B-q4f16_1-MLC',
  ].map((modelId) => ({
    modelId,
    sample: undefined,
    protocolOnly: modelId === 'Qwen3-1.7B-q4f16_1-MLC',
  })),
  {
    modelId: 'Qwen3-1.7B-q4f16_1-MLC',
    sample: { file: 'q3-91241.txt', code: 'code-999dca88' },
    protocolOnly: false,
  },
  {
    modelId: 'Qwen3-1.7B-q4f16_1-MLC',
    sample: { file: 'q3-36203.txt', code: 'code-18c583a1' },
    protocolOnly: true,
  },
]

for (const { modelId, sample, protocolOnly } of cases) {
  const behavior = protocolOnly
    ? 'preserves native output within a Harness turn limit'
    : modelId === 'Qwen3-0.6B-q4f16_1-MLC'
      ? 'matches native thinking and tool output'
      : 'calls a Harness tool and preserves native thinking and answer'
  test(
    `${modelId} ${behavior}` + (sample ? ` with recorded input ${sample.file}` : ''),
    modelId === 'Qwen3-0.6B-q4f16_1-MLC'
      ? {}
      : modelCheck(modelId, protocolOnly ? 'tool-protocol' : 'tools', 'thinking'),
    async ({ page }, info) => {
      const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
      const path = join(directory, sample?.file ?? `q3-${process.pid}.txt`)
      const code = sample?.code ?? 'code-' + crypto.randomUUID().slice(0, 8)
      const settings = sampling(modelId, true)
      const chunks: WebLLMChunk[][] = []
      let agent: Awaited<ReturnType<typeof createHarness>> | undefined
      let created = false

      try {
        await writeFile(path, `The secret code is ${code}.`, { flag: 'wx' })
        created = true
        await page.goto('http://127.0.0.1:4173/harness.html')
        await page.waitForFunction(() => Boolean(window.harness))
        await page.evaluate((id) => window.harness.open(id), modelId)
        await captureRequests(page)

        agent = await createHarness({
          model: new BrowserModel(page, {
            modelId,
            maxTokens: 4096,
            ...settings,
            request: { ...settings.request, seed: 7 },
            onChunk: (chunk) => chunks.at(-1)!.push(chunk),
          }),
          builtinTools: ['read'],
          builtinPlugins: [],
          session: false,
          skills: false,
          memory: false,
          contextManager: false,
          caching: false,
          backgroundTasks: false,
          printer: false,
        })
        let modelCalls = 0
        agent.addHook(BeforeModelCallEvent, () => {
          modelCalls++
          chunks.push([])
        })
        const result = await agent
          .invoke(
            `Use the read tool once with offset 1 to read ${path}. After it returns, answer with the secret code. Do not call the read tool again.`,
            {
              limits: { turns: 3 },
            },
          )
          .catch(async (error: unknown) => {
            if (
              modelId !== 'Qwen3-0.6B-q4f16_1-MLC' ||
              !(error instanceof Error) ||
              error.message !== 'Qwen ended inside a thinking block.'
            )
              throw error

            const request = (await page.evaluate(() => window.qwen3Requests)).at(-1)
            const raw = rawContent(chunks.at(-1) ?? [])
            const comparison = await compareNative(page, chunks)
            expect(comparison.native.map(rawContent)).toEqual(comparison.adapted.map(rawContent))
            await recordNative(info, 'incomplete-thinking', {
              ...comparison,
              modelId,
              request,
              raw,
              error: error.message,
            })
            expect(request).toBeDefined()
            expect(raw).toContain('<think>')
            expect(raw).not.toContain('</think>')
            info.annotations.push({
              type: 'native-output',
              description: 'Qwen3 0.6B stopped before closing its thinking block.',
            })
            return undefined
          })
        if (!result) return

        const comparison = await compareNative(page, chunks)
        expect(comparison.native.map(rawContent)).toEqual(comparison.adapted.map(rawContent))
        await recordNative(info, 'native-comparison', {
          code,
          path,
          stopReason: result.stopReason,
          modelCalls,
          messages: agent.messages.map((message) => message.toJSON()),
          ...comparison,
        })
        const toolResults = agent.messages
          .flatMap((message) => message.content)
          .filter((block) => block.type === 'toolResultBlock')
        expect(toolResults.length).toBeGreaterThan(0)
        expect(toolResults.some((block) => block.status === 'success')).toBe(true)
        expect(JSON.stringify(toolResults)).toContain(code)

        if (protocolOnly) {
          expect(modelCalls).toBe(comparison.requests.length)
          checkBoundedLoop(agent.messages, comparison, result.stopReason, modelId)
          return
        }

        expect(modelCalls).toBeGreaterThanOrEqual(2)
        expect(result.stopReason).toBe('endTurn')
        const content = result.lastMessage.content
        const text = content
          .filter((block) => block.type === 'textBlock')
          .map((block) => block.text)
          .join('')
        const thinking = agent.messages
          .filter((message) => message.role === 'assistant')
          .map((message) =>
            message.content
              .filter((block) => block.type === 'reasoningBlock')
              .map((block) => block.text ?? '')
              .join(''),
          )
        expect(text.trim()).not.toBe('')
        expect({ text, thinking }).toEqual(nativeOutput(chunks.flat(), modelId))
        expect(JSON.stringify(comparison.requests.at(-1))).toContain(code)
      } finally {
        try {
          await agent?.shutdown()
          await page.evaluate(() => window.harness?.close())
        } finally {
          if (created) await rm(path, { force: true })
        }
      }
    },
  )
}

test('Qwen3 preserves native tool order and arguments through Harness', async ({ page }, info) => {
  const modelId = 'Qwen3-4B-q4f16_1-MLC'
  const code = 'code-' + crypto.randomUUID().slice(0, 8)
  const receipt = 'receipt-' + crypto.randomUUID().slice(0, 8)
  const settings = sampling(modelId, true)
  const inputs: unknown[] = []
  const chunks: WebLLMChunk[][] = []
  const lookup = new FunctionTool({
    name: 'lookup',
    description: 'Return the code for a city.',
    inputSchema: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
    },
    callback: (input) => {
      inputs.push(input)
      return { code }
    },
  })
  const confirm = new FunctionTool({
    name: 'confirm',
    description: 'Confirm a code and return a receipt.',
    inputSchema: {
      type: 'object',
      properties: { code: { type: 'string' } },
      required: ['code'],
    },
    callback: (input) => {
      inputs.push(input)
      return { receipt }
    },
  })

  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  await captureRequests(page)
  let agent: Awaited<ReturnType<typeof createHarness>> | undefined

  try {
    agent = await createHarness({
      model: new BrowserModel(page, {
        modelId,
        maxTokens: 4096,
        ...settings,
        request: { ...settings.request, seed: 7 },
        onChunk: (chunk) => chunks.at(-1)!.push(chunk),
      }),
      tools: [lookup, confirm],
      builtinTools: [],
      builtinPlugins: [],
      session: false,
      skills: false,
      memory: false,
      contextManager: false,
      caching: false,
      backgroundTasks: false,
      printer: false,
    })
    let modelCalls = 0
    agent.addHook(BeforeModelCallEvent, () => {
      modelCalls++
      chunks.push([])
    })
    const result = await agent.invoke(
      'Call lookup for Seoul. After its result, call confirm with the returned code. ' +
        'Then tell me the receipt.',
      { limits: { turns: 4 } },
    )
    const comparison = await compareNative(page, chunks)
    expect(comparison.native.map(rawContent)).toEqual(comparison.adapted.map(rawContent))
    await recordNative(info, 'native-comparison', {
      code,
      receipt,
      inputs,
      messages: agent.messages.map((message) => message.toJSON()),
      ...comparison,
    })
    const toolResults = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolResultBlock')

    const calls = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolUseBlock')
    if (modelCalls === 2 && calls.length === 2) {
      expect(inputs).toEqual(calls.map((call) => call.input))
      expect(inputs[0]).toEqual({ city: 'Seoul' })
      expect(calls.map((call) => call.name)).toEqual(['lookup', 'confirm'])
      expect(toolResults).toHaveLength(2)
      expect(JSON.stringify(toolResults)).toContain(code)
      expect(JSON.stringify(toolResults)).toContain(receipt)
      expect(result.stopReason).toBe('endTurn')
      expect(result.lastMessage.content).toContainEqual(
        expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(receipt) }),
      )
      info.annotations.push({
        type: 'native-output',
        description:
          'Qwen3 4B emitted dependent tool calls in one response; native output matched.',
      })
      return
    }

    expect(inputs).toEqual([{ city: 'Seoul' }, { code }])
    expect(modelCalls).toBe(3)
    expect(toolResults).toHaveLength(2)
    expect(JSON.stringify(toolResults)).toContain(code)
    expect(JSON.stringify(toolResults)).toContain(receipt)
    expect(result.stopReason).toBe('endTurn')
    expect(result.lastMessage.content).toContainEqual(
      expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(receipt) }),
    )
  } finally {
    try {
      await agent?.shutdown()
    } finally {
      await page.evaluate(() => window.harness?.close())
    }
  }
})

async function captureRequests(page: Page) {
  await page.evaluate(() =>
    window.webllmBridge.withEngine((engine) => {
      window.qwen3Requests = []
      const endpoint = engine.chat.completions
      endpoint.create = new Proxy(endpoint.create, {
        apply(target, self, args) {
          window.qwen3Requests.push(structuredClone(args[0]))
          return Reflect.apply(target, self, args)
        },
      })
    }),
  )
}

async function compareNative(page: Page, adapted: WebLLMChunk[][]) {
  const requests = await page.evaluate(() => window.qwen3Requests)
  const native = await replayNative(page, requests)
  return { requests, adapted, native }
}

function checkBoundedLoop(
  messages: readonly Message[],
  { requests, native }: Awaited<ReturnType<typeof compareNative>>,
  stopReason: string,
  modelId: string,
) {
  expect(['endTurn', 'limitTurns']).toContain(stopReason)
  expect(requests.length).toBeGreaterThan(0)
  expect(requests.length).toBeLessThanOrEqual(3)
  if (stopReason === 'limitTurns') expect(requests).toHaveLength(3)
  const assistants = messages.filter((message) => message.role === 'assistant')
  expect(assistants).toHaveLength(native.length)
  const thinking = assistants.map((message) =>
    message.content
      .filter((block) => block.type === 'reasoningBlock')
      .map((block) => block.text ?? '')
      .join(''),
  )
  expect(thinking).toEqual(nativeOutput(native.flat(), modelId).thinking)
  const results = messages
    .flatMap((message) => message.content)
    .filter((block) => block.type === 'toolResultBlock')
  const byId = new Map(results.map((result) => [result.toolUseId, result]))
  let callCount = 0
  for (const [index, message] of assistants.entries()) {
    const content = rawContent(native[index]!).replace(/<think>[\s\S]*?<\/think>/g, '')
    const tool = /<tool_call>([\s\S]*?)<\/tool_call>/g
    const expectedCalls = [...content.matchAll(tool)].map((match) => {
      const call: Record<string, unknown> = JSON.parse(match[1]!)
      return { name: call.name, arguments: call.arguments }
    })
    const calls = message.content.filter((block) => block.type === 'toolUseBlock')
    expect(calls.map((call) => ({ name: call.name, arguments: call.input }))).toEqual(expectedCalls)
    const text = message.content
      .filter((block) => block.type === 'textBlock')
      .map((block) => block.text)
      .join('')
    const expectedText = content.replace(tool, '')
    // Strands omits blank text blocks when aggregating a model response.
    expect(text).toBe(expectedText.trim() ? expectedText : '')
    callCount += calls.length
    for (const call of calls) {
      const result = byId.get(call.toolUseId)
      expect(result).toBeDefined()
      const next = requests[index + 1]
      if (!next) continue
      for (const block of result!.content) {
        if (block.type !== 'textBlock') continue
        expect(
          next.messages.some(
            (message) =>
              typeof message.content === 'string' && message.content.includes(block.text),
          ),
        ).toBe(true)
      }
    }
  }
  expect(results).toHaveLength(callCount)
  expect(byId.size).toBe(callCount)
}

function rawContent(chunks: WebLLMChunk[]) {
  return chunks
    .map((chunk) => {
      const choice = chunk.choices?.[0]
      if (!choice) return ''
      return 'delta' in choice ? (choice.delta.content ?? '') : (choice.text ?? '')
    })
    .join('')
}

declare global {
  interface Window {
    qwen3Requests: ChatCompletionRequestStreaming[]
  }
}
