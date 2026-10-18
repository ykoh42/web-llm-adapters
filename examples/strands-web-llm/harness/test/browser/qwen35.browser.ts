import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { BeforeModelCallEvent } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel } from 'strands-web-llm'

for (const modelId of [
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-9B-q4f16_1-MLC',
]) {
  test(`${modelId} thinking calls Harness read and answers from its result`, async ({ page }) => {
    const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
    const path = join(directory, `q35-${process.pid}.txt`)
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
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
          request: { extra_body: { enable_thinking: true } },
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
      expect(
        agent.messages
          .flatMap((message) => message.content)
          .some((block) => block.type === 'reasoningBlock' && block.text),
      ).toBe(true)
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
      }
    }
  })
}
