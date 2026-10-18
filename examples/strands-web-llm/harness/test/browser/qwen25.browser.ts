import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { BeforeModelCallEvent } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'

for (const modelId of [
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
]) {
  test(`${modelId} calls Harness read and answers from its result`, async ({ page }, info) => {
    const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
    const path = join(directory, `q25-${process.pid}.txt`)
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
    const chunks: WebLLMChunk[] = []
    let agent: Awaited<ReturnType<typeof createHarness>> | undefined

    try {
      await writeFile(path, `The secret code is ${code}.`)
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
        await rm(path, { force: true })
        await info.attach('native-chunks', {
          body: JSON.stringify(chunks),
          contentType: 'application/json',
        })
      }
    }
  })
}
