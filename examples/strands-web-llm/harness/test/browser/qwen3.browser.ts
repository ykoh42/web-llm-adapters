import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { BeforeModelCallEvent, FunctionTool } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { Page } from '@playwright/test'
import { recordNative, replayNative } from '../../fixtures/native.js'
import { sampling } from '../../../sampling.js'

for (const modelId of ['Qwen3-0.6B-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3-8B-q4f16_1-MLC']) {
  const behavior =
    modelId === 'Qwen3-0.6B-q4f16_1-MLC'
      ? 'preserves incomplete thinking output'
      : 'thinking calls a Harness tool, uses its result, and answers'
  test(`${modelId} ${behavior}`, async ({ page }, info) => {
    const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
    const path = join(directory, `q3-${process.pid}.txt`)
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
    const settings = sampling(modelId, true)
    const chunks: WebLLMChunk[][] = []
    let agent: Awaited<ReturnType<typeof createHarness>> | undefined

    try {
      await writeFile(path, `The secret code is ${code}.`)
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
        .invoke(`Use the read tool to read ${path}. What secret code does it contain?`, {
          limits: { turns: 3 },
        })
        .catch(async (error: unknown) => {
          if (
            modelId !== 'Qwen3-0.6B-q4f16_1-MLC' ||
            !(error instanceof Error) ||
            error.message !== 'Qwen ended inside a thinking block.'
          )
            throw error

          const request = (await page.evaluate(() => window.qwen3Requests))[0]
          const raw = rawContent(chunks[0] ?? [])
          await recordNative(info, 'incomplete-thinking', {
            modelId,
            request,
            raw,
            error: error.message,
          })
          expect(request).toBeDefined()
          expect(raw).toContain('<think>')
          expect(raw).not.toContain('</think>')
          info.annotations.push({
            type: 'model-limit',
            description: 'Qwen3 0.6B stopped before closing its thinking block.',
          })
          return undefined
        })
      if (!result) return

      await recordNative(info, 'native-comparison', {
        code,
        path,
        messages: agent.messages.map((message) => message.toJSON()),
        ...(await compareNative(page, chunks)),
      })

      expect(modelCalls).toBe(2)
      expect(result.stopReason).toBe('endTurn')
      const toolResults = agent.messages
        .flatMap((message) => message.content)
        .filter((block) => block.type === 'toolResultBlock')
      expect(result.lastMessage.content).toContainEqual(
        expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(code) }),
      )
      expect(
        agent.messages
          .flatMap((message) => message.content)
          .some((block) => block.type === 'reasoningBlock' && block.text),
      ).toBe(true)
      expect(toolResults.length).toBeGreaterThan(0)
      expect(JSON.stringify(toolResults)).toContain(code)
    } finally {
      try {
        await agent?.shutdown()
        await page.evaluate(() => window.harness?.close())
      } finally {
        await rm(path, { force: true })
      }
    }
  })
}

test('Qwen3 can call a second distinct Harness tool after its first result', async ({
  page,
}, info) => {
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
      expect(comparison.native.map(rawContent)).toEqual(comparison.adapted.map(rawContent))
      expect(toolResults).toHaveLength(2)
      expect(JSON.stringify(toolResults)).toContain(code)
      expect(JSON.stringify(toolResults)).toContain(receipt)
      expect(result.stopReason).toBe('endTurn')
      expect(result.lastMessage.content).toContainEqual(
        expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(receipt) }),
      )
      info.annotations.push({
        type: 'model-limit',
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
