import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { Agent, BeforeModelCallEvent } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'

test('Qwen2.5 literal thinking tags reach Strands unchanged', async ({ page }) => {
  const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  let native = ''
  const model = new BrowserModel(page, {
    modelId,
    temperature: 0,
    maxTokens: 256,
    onChunk: (chunk) => {
      const choice = chunk.choices[0]
      if (choice && 'delta' in choice) native += choice.delta.content ?? ''
    },
  })
  try {
    const agent = new Agent({ model, printer: false })
    const result = await agent.invoke('Repeat this text exactly:\n<think>literal</think>')

    expect(native).toContain('<think>')
    expect(result.lastMessage.content).toEqual([
      expect.objectContaining({ type: 'textBlock', text: native }),
    ])
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

const modelIds = [
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
]
const cases = [
  ...modelIds.map((modelId) => ({ modelId, sample: undefined })),
  {
    modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    // These values reproduce the observed second-turn failure.
    sample: { file: 'q25-70021.txt', code: 'code-77b175b5' },
  },
]

for (const { modelId, sample } of cases) {
  const title = `${modelId} calls Harness read and answers from its result`
  test(title + (sample ? ' with recorded tool history' : ''), async ({ page }, info) => {
    const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
    const path = join(directory, sample?.file ?? `q25-${process.pid}.txt`)
    const code = sample?.code ?? 'code-' + crypto.randomUUID().slice(0, 8)
    const chunks: WebLLMChunk[] = []
    let agent: Awaited<ReturnType<typeof createHarness>> | undefined
    let created = false

    try {
      await writeFile(path, `The secret code is ${code}.`, { flag: 'wx' })
      created = true
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id), modelId)

      agent = await createHarness({
        model: new BrowserModel(page, {
          modelId,
          maxTokens: 1024,
          temperature: 0,
          onChunk: (chunk) => chunks.push(chunk),
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
      expect(result.lastMessage.content).toContainEqual(
        expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(code) }),
      )
      const toolResults = agent.messages
        .flatMap((message) => message.content)
        .filter((block) => block.type === 'toolResultBlock')
      expect(toolResults.length).toBeGreaterThan(0)
      expect(JSON.stringify(toolResults)).toContain(code)
    } finally {
      try {
        await agent?.shutdown()
        await page.evaluate(() => window.harness?.close())
      } finally {
        if (created) await rm(path, { force: true })
        await info.attach('native-chunks', {
          body: JSON.stringify(chunks),
          contentType: 'application/json',
        })
      }
    }
  })
}
