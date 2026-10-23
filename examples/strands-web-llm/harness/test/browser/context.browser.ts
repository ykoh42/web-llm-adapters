import { writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import {
  Message,
  TextBlock,
  ReasoningBlock,
  FunctionTool,
  AfterModelCallEvent,
} from '@strands-agents/sdk'
import { test, expect } from '../../../fixtures/browser.js'
import { openHarness, useTool, finalText } from '../../fixtures/harness.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'

test('custom config readers align token estimates with real Gemma Harness requests', async ({
  page,
}) => {
  const harness = await openHarness(page)
  const config = harness.model.getConfig()
  class CurrentModel extends BrowserModel {
    override getConfig() {
      return structuredClone(config)
    }
  }
  const model = new CurrentModel(page, { modelId: 'initial-model' })
  const history = [
    new Message({ role: 'user', content: [new TextBlock('Earlier question.')] }),
    new Message({
      role: 'assistant',
      content: [
        new ReasoningBlock({ text: 'Archived plan. '.repeat(2000) }),
        new TextBlock('Done.'),
      ],
    }),
    new Message({ role: 'user', content: [new TextBlock('Next question.')] }),
  ]
  try {
    expect(await model.countTokens(history)).toBe(await harness.model.countTokens(history))
    const code = crypto.randomUUID().slice(0, 8)
    const path = join(harness.dir, 'code.txt')
    await writeFile(path, `The secret code is ${code}.`)
    const agent = await harness.create({ model, messages: history })
    const previousMessages = agent.messages.length
    useTool(agent, 'read')
    const result = await agent.invoke(
      `Use read to read ${path}. Reply with the secret code from that file.`,
      { limits: { turns: 4 } },
    )
    expect(finalText(result)).toContain(code)
    expect(agent.messages.flatMap((message) => message.content)).toContainEqual(
      expect.objectContaining({ type: 'toolResultBlock', status: 'success' }),
    )
    expect(
      agent.messages
        .slice(previousMessages)
        .flatMap((message) => message.content)
        .some((block) => block.type === 'reasoningBlock' && block.text?.trim()),
    ).toBe(true)
    expect(await page.evaluate(() => window.gemmaPrompts)).not.toEqual([])
    expect((await page.evaluate(() => window.gemmaPrompts)).join('\n')).not.toContain(
      'Archived plan.',
    )
  } finally {
    await harness.close()
  }
})

for (const longRecent of [false, true]) {
  test(`default context summary preserves facts with ${longRecent ? 'long' : 'short'} recent messages`, async ({
    page,
  }) => {
    const harness = await openHarness(page)
    try {
      harness.model.updateConfig({ contextWindowLimit: 8192 })
      const code = 'audit-' + crypto.randomUUID().slice(0, 8)
      const messages = Array.from({ length: 12 }, (_, i) => {
        let text = longRecent
          ? 'These archived notes repeat a stable project fact.\n'.repeat(65)
          : 'We will use the saved audit label for the next check.'
        if (i === 0) text = 'We are reviewing old project notes.'
        if (i > 0 && i < 8)
          text =
            `The audit label is ${code}.\n` +
            'These archived notes repeat a stable project fact.\n'.repeat(65)
        return new Message({
          role: i % 2 ? 'assistant' : 'user',
          content: [new TextBlock(text)],
        })
      })
      const agent = await harness.create({ messages })
      const result = await agent.invoke(
        longRecent
          ? 'Reply with the word ready.'
          : 'Read the audit label from the summarized notes. Reply with its exact audit- value, including the full suffix.',
        { limits: { turns: 3 } },
      )
      const summaries = agent.messages
        .flatMap((m) => m.content)
        .filter((b) => b.type === 'textBlock' && b.text.includes('[Summarized:'))
      expect(JSON.stringify(summaries)).toContain(code)
      if (longRecent) expect(finalText(result)).toMatch(/ready/i)
      else expect(finalText(result)).toContain(code)
      expect(agent.messages.length).toBeLessThan(14)
      expect((await agent.contextManager?.stash?.list())?.length).toBeGreaterThan(0)
    } finally {
      await harness.close()
    }
  })
}

test('a real WebLLM context overflow reaches Harness and recovers by trimming history', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    // Deliberately overestimate the loaded window to exercise provider-error recovery.
    harness.model.updateConfig({ contextWindowLimit: 131072 })
    const messages = Array.from(
      // The default manager retries at most three times; keep this overflow recoverable.
      { length: 20 },
      (_, i) =>
        new Message({
          role: i % 2 ? 'assistant' : 'user',
          content: [
            new TextBlock(
              i === 0 ? 'Archived transcript.' : 'Old unrelated record number 1234. '.repeat(120),
            ),
          ],
        }),
    )
    const agent = await harness.create({ messages })
    const errors: string[] = []
    agent.addHook(AfterModelCallEvent, (event) => {
      if (event.error) errors.push(event.error.name)
    })
    const result = await agent.invoke('Reply with the word ready.', { limits: { turns: 3 } })
    expect(errors).toContain('ContextWindowOverflowError')
    expect(result.stopReason).toBe('endTurn')
    expect(result.lastMessage.content).toContainEqual(
      expect.objectContaining({ type: 'textBlock', text: expect.stringMatching(/ready/i) }),
    )
    expect(agent.messages.length).toBeLessThan(22)
  } finally {
    await harness.close()
  }
})

for (const { choice, repeated } of [
  { choice: 'auto', repeated: false },
  { choice: 'auto', repeated: true },
  { choice: 'named', repeated: false },
  { choice: 'named', repeated: true },
]) {
  test(`${choice} thinking-enabled offloading retrieves ${repeated ? 'repeated' : 'unique'} notes`, async ({
    page,
  }, info) => {
    const harness = await openHarness(page)
    const chunks: WebLLMChunk[] = []
    try {
      const code = 'archive-' + Math.floor(Math.random() * 1_000_000)
      const lines = Array.from({ length: 500 }, (_, i) => {
        if (i === 250) return `The verification code is ${code}.`
        if (repeated) return `Archive line ${i}: ordinary project notes without a code.`
        return `Project record ${i}: reviewed task ${crypto.randomUUID()}.`
      })
      const archive = new FunctionTool({
        name: 'archive',
        description: 'Read the archived project notes.',
        inputSchema: { type: 'object', properties: {} },
        callback: () => lines.join('\n'),
      })
      const model = new BrowserModel(page, {
        ...harness.model.getConfig(),
        onChunk: (chunk) => chunks.push(chunk),
      })
      const agent = await harness.create({ model, tools: [archive] })
      if (choice === 'named') useTool(agent, 'archive')
      const received = await agent.invoke(
        repeated
          ? 'Call archive in the foreground. Acknowledge receiving the notes without extracting a code yet.'
          : 'Call archive in the foreground. Keep its result for a later question.',
        { limits: { turns: 4 } },
      )
      expect(received.stopReason).toBe('endTurn')
      expect(agent.messages.flatMap((message) => message.content)).toContainEqual(
        expect.objectContaining({ type: 'toolUseBlock', name: 'archive' }),
      )
      expect(
        received.lastMessage.content.some(
          (block) => block.type === 'reasoningBlock' && block.text?.trim(),
        ),
      ).toBe(true)
      if (repeated) expect(finalText(received).trim().length).toBeGreaterThan(0)
      const message = agent.messages.find((m) =>
        m.content.some(
          (b) => b.type === 'toolResultBlock' && JSON.stringify(b).includes('[Truncated:'),
        ),
      )
      if (!message) throw new Error('The default context manager did not offload the result')
      const index = message.content.findIndex(
        (b) => b.type === 'toolResultBlock' && JSON.stringify(b).includes('[Truncated:'),
      )
      const block = message.content[index]!
      expect(JSON.stringify(block)).not.toContain(code)
      const stash = agent.contextManager?.stash
      if (!stash) throw new Error('The default context stash is missing')
      const reference = stash.refsFor(block, message, index)[0]!
      expect(JSON.stringify(await stash.retrieve(reference))).toContain(code)
      if (choice === 'named') useTool(agent, 'retrieve_context')
      const result = await agent.invoke(
        `Use retrieve_context with reference "${reference}", pattern "verification", and context_lines 0. Tell me the verification code from that result.`,
        { limits: { turns: 4 } },
      )
      expect(finalText(result)).toContain(code)
      expect(
        await page.evaluate(() =>
          window.gemmaPrompts.every((prompt) => prompt.includes('<|think|>')),
        ),
      ).toBe(true)
      expect(agent.messages.flatMap((m) => m.content)).toContainEqual(
        expect.objectContaining({ type: 'toolUseBlock', name: 'retrieve_context' }),
      )
    } catch (error) {
      const path = info.outputPath('native-chunks.json')
      await writeFile(
        path,
        JSON.stringify({ prompts: await page.evaluate(() => window.gemmaPrompts), chunks }),
      )
      await info.attach('native-chunks', { path, contentType: 'application/json' })
      throw error
    } finally {
      await harness.close()
    }
  })
}
