import { modelCheck } from '../../../../../scripts/release/support.js'
import { createHarness } from '@strands-agents/harness'
import { ImageBlock, TextBlock } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel } from 'strands-web-llm'
import { finalText } from '../../fixtures/harness.js'

test(
  'Phi vision reads a Strands ImageBlock through Harness',
  modelCheck('Phi-3.5-vision-instruct-q4f16_1-MLC', 'image'),
  async ({ page }) => {
    const modelId = 'Phi-3.5-vision-instruct-q4f16_1-MLC'
    const color = ['red', 'green', 'blue'][Math.floor(Math.random() * 3)]!
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id, 8192), modelId)
    let agent: Awaited<ReturnType<typeof createHarness>> | undefined

    try {
      const image = await page.evaluate((color) => {
        const canvas = document.createElement('canvas')
        canvas.width = canvas.height = 224
        const context = canvas.getContext('2d')!
        context.fillStyle = color
        context.fillRect(0, 0, canvas.width, canvas.height)
        return canvas.toDataURL('image/png').split(',')[1]!
      }, color)
      const bytes = Uint8Array.from(atob(image), (character) => character.charCodeAt(0))

      agent = await createHarness({
        model: new BrowserModel(page, {
          modelId,
          contextWindowLimit: 8192,
          maxTokens: 64,
          temperature: 0,
        }),
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
      const result = await agent.invoke([
        new TextBlock('What color fills the image? Reply with one color.'),
        new ImageBlock({ format: 'png', source: { bytes } }),
      ])

      expect(finalText(result).toLowerCase()).toContain(color)
      expect(agent.messages[0]?.content.some((block) => block.type === 'imageBlock')).toBe(true)
    } finally {
      try {
        await agent?.shutdown()
      } finally {
        await page.evaluate(() => window.harness.close())
      }
    }
  },
)
