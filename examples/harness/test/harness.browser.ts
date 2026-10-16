import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness, type HarnessAgentOptions } from '@strands-agents/harness'
import { AfterToolsEvent } from '@strands-agents/sdk'
import { test, expect } from '../../../test/browser/helpers.js'
import { BrowserModel } from '../browser-model.js'

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

test('Harness reads a Node.js file using its built-in tool selected by WebLLM', async ({
  page,
}) => {
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
      event.endTurn = 'Read finished.'
    })
    expect(String(await agent.invoke(`Use the read tool to read ${path}.`))).toContain(
      'Read finished.',
    )
    const results = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolResultBlock')
    expect(JSON.stringify(results)).toContain('pineapple-42')
  } finally {
    await page.evaluate(() => window.harness.close())
    await rm(directory, { recursive: true, force: true })
  }
})
