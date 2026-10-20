import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness, type HarnessAgentOptions } from '@strands-agents/harness'
import { AfterToolsEvent, BeforeModelCallEvent, FunctionTool, TextBlock } from '@strands-agents/sdk'
import { test, expect } from '../../../fixtures/browser.js'
import { BrowserModel } from 'strands-web-llm'

const defaults: HarnessAgentOptions = {
  builtinTools: [],
  builtinPlugins: [],
  session: false,
  skills: false,
  memory: false,
  contextManager: false,
  caching: false,
  backgroundTasks: false,
  printer: false,
}

test('Harness streams browser WebLLM output and keeps conversation history', async ({ page }) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    const agent = await createHarness({
      ...defaults,
      model: new BrowserModel(page, { modelId, maxTokens: 128, temperature: 0 }),
      systemPrompt: 'Answer briefly.',
    })
    await agent.invoke('Remember my codeword: pineapple. Say OK.')
    let text = ''
    for await (const update of agent.stream('What is my codeword?')) {
      if (update.type !== 'modelStreamUpdateEvent') continue
      const event = update.event
      if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
        text += event.delta.text
      }
    }
    expect(text.toLowerCase()).toContain('pineapple')
    expect(agent.messages).toHaveLength(4)
    const controller = new AbortController()
    try {
      for await (const update of agent.stream('Count from 1 to 1000.', {
        cancelSignal: controller.signal,
      })) {
        if (
          update.type === 'modelStreamUpdateEvent' &&
          update.event.type === 'modelContentBlockDeltaEvent'
        )
          controller.abort()
      }
    } catch (error) {
      if (!controller.signal.aborted) throw error
    }
    expect(controller.signal.aborted).toBe(true)
    expect(String(await agent.invoke('Say hello.'))).toMatch(/hello|hi/i)
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

for (const thinking of [false, true]) {
  test(`Gemma nested tools with thinking ${thinking} use a result in the next call`, async ({
    page,
  }) => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
    const receipt = 'receipt-' + crypto.randomUUID().slice(0, 8)
    const inputs: unknown[] = []
    const lookup = new FunctionTool({
      name: 'lookup',
      description: 'Get the code for a city and options.',
      inputSchema: {
        type: 'object',
        properties: {
          city: { type: 'string' },
          options: {
            type: 'object',
            properties: {
              days: { type: 'array', items: { type: 'number' } },
              active: { type: 'boolean' },
            },
            required: ['days', 'active'],
          },
        },
        required: ['city', 'options'],
      },
      callback: (input) => {
        inputs.push(input)
        return { code }
      },
    })
    const confirm = new FunctionTool({
      name: 'confirm',
      description: 'Confirm a code returned by lookup and get a receipt.',
      inputSchema: {
        type: 'object',
        properties: { code: { type: 'string' } },
        required: ['code'],
      },
      callback: (input) => {
        inputs.push(input)
        expect(input).toEqual({ code })
        return { receipt }
      },
    })
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      const agent = await createHarness({
        ...defaults,
        model: new BrowserModel(page, {
          modelId,
          maxTokens: 1024,
          temperature: 0,
          request: { extra_body: { enable_thinking: thinking } },
        }),
        tools: [lookup, confirm],
      })
      let modelCalls = 0
      agent.addHook(BeforeModelCallEvent, () => {
        modelCalls++
      })
      const result = await agent.invoke(
        'Call lookup with city Seoul and options {days:[1,2],active:true}. ' +
          'Wait for its code, then call confirm with that code. Finally tell me the receipt.',
        { limits: { turns: 4 } },
      )
      console.log('Gemma receipt:', String(result))
      expect(inputs).toEqual([{ city: 'Seoul', options: { days: [1, 2], active: true } }, { code }])
      expect(modelCalls).toBe(3)
      expect(result.stopReason).toBe('endTurn')
      expect(result.lastMessage.content).toContainEqual(
        expect.objectContaining({
          type: 'textBlock',
          text: expect.stringContaining(receipt),
        }),
      )
      if (thinking)
        expect(
          agent.messages
            .flatMap((message) => message.content)
            .some((block) => block.type === 'reasoningBlock' && block.text),
        ).toBe(true)
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}

test('Hermes runs one Harness read batch without an assistant answer', async ({ page }) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  const directory = await mkdtemp(join(tmpdir(), 'webllm-harness-'))
  const path = join(directory, 'note.txt')
  await writeFile(path, 'The test codeword is pineapple-42.')
  try {
    const agent = await createHarness({
      ...defaults,
      model: new BrowserModel(page, { modelId, maxTokens: 256, temperature: 0 }),
      systemPrompt: '',
      builtinTools: ['read'],
    })
    agent.addHook(AfterToolsEvent, (event) => {
      event.endTurn = [new TextBlock('')]
    })
    const result = await agent.invoke(`Use the read tool to read ${path}.`)
    expect(String(result)).toBe('')
    const results = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolResultBlock')
    expect(results).toHaveLength(1)
    expect(JSON.stringify(results)).toContain('pineapple-42')
  } finally {
    await page.evaluate(() => window.harness.close())
    await rm(directory, { recursive: true, force: true })
  }
})

test('Gemma selects Harness read, receives its result and writes the final answer', async ({
  page,
}) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  const directory = await mkdtemp(join(tmpdir(), 'gemma-harness-'))
  const path = join(directory, 'note.txt')
  const code = 'lime-' + crypto.randomUUID().slice(0, 8)
  try {
    await writeFile(
      path,
      `The secret code is ${code}.\nSource sample: const quote = '<|"|>'; // <|turn> <turn|>`,
    )
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    const agent = await createHarness({
      ...defaults,
      model: new BrowserModel(page, { modelId, maxTokens: 512, temperature: 0 }),
      builtinTools: ['read'],
    })
    let modelCalls = 0
    agent.addHook(BeforeModelCallEvent, () => {
      modelCalls++
    })
    const result = await agent.invoke(
      `Read ${path} using the read tool. What is the secret code in it?`,
      { limits: { turns: 3 } },
    )
    console.log('Gemma final answer:', String(result))
    expect(result.stopReason).toBe('endTurn')
    expect(String(result)).toContain(code)
    expect(modelCalls).toBe(2)
    const results = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolResultBlock')
    expect(results).toHaveLength(1)
    expect(results[0]?.status).toBe('success')
    expect(JSON.stringify(results)).toContain(code)
    const next = await agent.invoke('What was the secret code? Reply with only the code.', {
      limits: { turns: 3 },
    })
    expect(String(next)).toContain(code)
    expect(modelCalls).toBe(3)
  } finally {
    try {
      await page.evaluate(() => window.harness?.close())
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  }
})
