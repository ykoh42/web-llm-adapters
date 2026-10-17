import { ImageBlock, Message, TextBlock } from '@strands-agents/sdk'
import { writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { test, expect } from '../../../fixtures/browser.js'
import { BrowserModel } from 'strands-web-llm'
import { openHarness, useTool, finalText } from '../../fixtures/harness.js'
import { pdf } from '../../fixtures/pdf.js'

test('the adapter keeps concurrent OCR tool results, thinking and answers separate on one Gemma', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const codes = ['31', '74'].map(
      (prefix) => prefix + String(Math.floor(Math.random() * 10000)).padStart(4, '0'),
    )
    const paths: string[] = []
    for (const [i, format] of ['png', 'jpeg'].entries()) {
      const path = join(harness.dir, 'note-' + i + '.' + format)
      const data = await page.evaluate(
        ({ code, format }) => {
          const canvas = document.createElement('canvas')
          canvas.width = 1200
          canvas.height = 360
          const context = canvas.getContext('2d')!
          context.fillStyle = 'white'
          context.fillRect(0, 0, canvas.width, canvas.height)
          context.fillStyle = 'black'
          context.font = '48px sans-serif'
          context.fillText('Verification code: ' + code, 50, 110)
          context.fillText('확인 번호: ' + code, 50, 230)
          return canvas.toDataURL('image/' + format).split(',')[1]!
        },
        { code: codes[i]!, format },
      )
      await writeFile(path, Buffer.from(data, 'base64'))
      paths.push(path)
    }
    const agents = await Promise.all(paths.map(() => harness.create()))
    for (const agent of agents) useTool(agent, 'read')
    const results = await Promise.all(
      agents.map((agent, i) =>
        agent.invoke(
          `Read the text in ${paths[i]}. Reply with only the verification code from the tool result.`,
          { limits: { turns: 3 } },
        ),
      ),
    )
    const prompts = await page.evaluate(() => window.gemmaPrompts)
    for (const [i, result] of results.entries()) {
      const blocks = agents[i]!.messages.flatMap((message) => message.content)
      const saved = blocks.find((block) => block.type === 'toolResultBlock')
      expect(saved, JSON.stringify(saved)).toMatchObject({ status: 'success' })
      expect(JSON.stringify(saved)).toContain(codes[i])
      expect(JSON.stringify(saved)).toContain('확인')
      expect(
        prompts.some((prompt) => prompt.includes(codes[i]!) && !prompt.includes(codes[1 - i]!)),
      ).toBe(true)
      expect(blocks.some((block) => block.type === 'reasoningBlock' && block.text)).toBe(true)
      expect(finalText(result)).toContain(codes[i])
      expect(finalText(result)).not.toContain(codes[1 - i])
    }
  } finally {
    await harness.close()
  }
})

test('the shipped Gemma artifact rejects a real PNG through its native image API', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const url = await page.evaluate(() => {
      const canvas = document.createElement('canvas')
      canvas.width = canvas.height = 64
      const context = canvas.getContext('2d')!
      context.fillStyle = 'red'
      context.fillRect(0, 0, 64, 64)
      return canvas.toDataURL('image/png')
    })
    const model = new BrowserModel(page, {
      modelId: harness.model.getConfig().modelId,
      maxTokens: 64,
      temperature: 0,
    })
    await expect(async () => {
      for await (const event of model.stream([
        new Message({
          role: 'user',
          content: [
            new TextBlock('What color is this image?'),
            new ImageBlock({ format: 'png', source: { url } }),
          ],
        }),
      ]))
        void event
    }).rejects.toThrow(/image.*support|support.*image/i)
  } finally {
    await harness.close()
  }
})

test('Gemma reads a real PDF file through Harness and answers from its result', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const path = join(harness.dir, 'note.pdf')
    const code = 'pdf-' + Math.floor(Math.random() * 1_000_000)
    await writeFile(path, pdf(`The verification code is ${code}.`))
    const agent = await harness.create()
    useTool(agent, 'read')
    const result = await agent.invoke(
      `Read ${path}. Reply with only the verification code from its contents.`,
      { limits: { turns: 3 } },
    )
    const saved = agent.messages
      .flatMap((message) => message.content)
      .find((block) => block.type === 'toolResultBlock')
    expect(saved, JSON.stringify(saved)).toMatchObject({ status: 'success' })
    expect(JSON.stringify(saved)).toContain(code)
    expect(finalText(result)).toContain(code)
  } finally {
    await harness.close()
  }
})

test('Gemma uses Harness Exa web_search and answers with an actual search source', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const agent = await harness.create()
    useTool(agent, 'web_search')
    const result = await agent.invoke(
      'Use web_search to find the official harness-sdk repository in the strands-agents GitHub organization. Reply with only the exact repository URL given in the search results.',
      { limits: { turns: 3 } },
    )
    const saved = agent.messages
      .flatMap((message) => message.content)
      .find((block) => block.type === 'toolResultBlock')
    expect(saved, JSON.stringify(saved)).toMatchObject({ status: 'success' })
    expect(JSON.stringify(saved)).toContain('github.com/strands-agents/harness-sdk')
    expect(finalText(result)).toContain('github.com/strands-agents/harness-sdk')
  } finally {
    await harness.close()
  }
})
