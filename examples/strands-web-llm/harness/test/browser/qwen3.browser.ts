import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { BeforeModelCallEvent, FunctionTool } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel } from 'strands-web-llm'
import { sampling } from '../../../shared/sampling.js'

for (const modelId of ['Qwen3-0.6B-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3-8B-q4f16_1-MLC']) {
  test(`${modelId} thinking calls a Harness tool, uses its result, and answers`, async ({
    page,
  }) => {
    const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
    const path = join(directory, `q3-${process.pid}.txt`)
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
    const settings = sampling(modelId, true)
    let agent: Awaited<ReturnType<typeof createHarness>> | undefined

    try {
      await writeFile(path, `The secret code is ${code}.`)
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id), modelId)

      agent = await createHarness({
        model: new BrowserModel(page, {
          modelId,
          maxTokens: 4096,
          ...settings,
          request: { ...settings.request, seed: 7 },
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
      })
      const result = await agent.invoke(
        `Use the read tool to read ${path}. What secret code does it contain?`,
        { limits: { turns: 3 } },
      )

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

test('Qwen3 can call a second distinct Harness tool after its first result', async ({ page }) => {
  const modelId = 'Qwen3-4B-q4f16_1-MLC'
  const code = 'code-' + crypto.randomUUID().slice(0, 8)
  const receipt = 'receipt-' + crypto.randomUUID().slice(0, 8)
  const settings = sampling(modelId, true)
  const inputs: unknown[] = []
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
  let agent: Awaited<ReturnType<typeof createHarness>> | undefined

  try {
    agent = await createHarness({
      model: new BrowserModel(page, {
        modelId,
        maxTokens: 4096,
        ...settings,
        request: { ...settings.request, seed: 7 },
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
    })
    const result = await agent.invoke(
      'Call lookup for Seoul. After its result, call confirm with the returned code. ' +
        'Then tell me the receipt.',
      { limits: { turns: 4 } },
    )
    const toolResults = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolResultBlock')

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
