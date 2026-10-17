import { TextBlock } from '@strands-agents/sdk'
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { test, expect } from '../../test/browser.js'
import { openHarness, useTool, finalText } from './setup.js'
import { createMemory } from '../memory.js'

test('default sessions and memory save and recall a real Gemma conversation', async ({ page }) => {
  const harness = await openHarness(page)
  const code = 'build-' + crypto.randomUUID().slice(0, 8)
  const session = { id: 'first', dir: join(harness.dir, 'sessions') }
  try {
    const count = await harness.model.countTokens([], { systemPrompt: 'Count this text.' })
    expect(
      await harness.model.countTokens([], { systemPrompt: [new TextBlock('Count this text.')] }),
    ).toBe(count)
    const agent = await harness.create({ session })
    const text = `Our project uses release tag ${code}. Keep this release tag for future conversations. Reply with the tag.`
    const result = await agent.invoke(text, { limits: { turns: 3 } })
    expect(result.lastMessage.content).toContainEqual(
      expect.objectContaining({
        type: 'textBlock',
        text: expect.stringContaining(code),
      }),
    )
    await agent.shutdown()
    const files = await readdir(join(harness.dir, 'memory'))
    const memory = await Promise.all(
      files
        .filter((file) => file.endsWith('.md'))
        .map((file) => readFile(join(harness.dir, 'memory', file), 'utf8')),
    )
    expect(memory.join('\n')).toContain(code)
    expect(memory.join('\n')).not.toContain('Reply with the tag')

    const restored = await harness.create({ session })
    const reply = await restored.invoke(
      'What release tag does our project use? Reply with only that tag.',
      {
        limits: { turns: 3 },
      },
    )
    expect(restored.messages[0]?.content).toContainEqual(
      expect.objectContaining({ type: 'textBlock', text }),
    )
    expect(restored.messages.length).toBeGreaterThan(2)
    expect(finalText(reply)).toContain(code)
    await restored.shutdown()

    // A new session must recall extracted memory without receiving the old conversation.
    const store = createMemory(
      page,
      harness.model.getConfig().modelId,
      16384,
      join(harness.dir, 'memory'),
    )
    expect(JSON.stringify(await store.search('release tag'))).toContain(code)
    const fresh = await harness.create({ memory: { stores: [store] } })
    const start = await page.evaluate(() => window.gemmaPrompts.length)
    useTool(fresh, 'search_memory')
    const recall = await fresh.invoke(
      'Use search_memory with query "release tag". Reply with only our project release tag from that result.',
      {
        limits: { turns: 3 },
      },
    )
    expect(fresh.messages[0]?.content).not.toContainEqual(expect.objectContaining({ text }))
    expect((await page.evaluate(() => window.gemmaPrompts))[start]).toContain(code)
    const call = fresh.messages
      .flatMap((message) => message.content)
      .find((block) => block.type === 'toolUseBlock' && block.name === 'search_memory')
    if (!call || call.type !== 'toolUseBlock') throw new Error('Gemma did not call search_memory')
    const saved = fresh.messages
      .flatMap((message) => message.content)
      .find((block) => block.type === 'toolResultBlock' && block.toolUseId === call.toolUseId)
    expect(saved).toMatchObject({ type: 'toolResultBlock', status: 'success' })
    expect(JSON.stringify(saved)).toContain(code)
    expect(finalText(recall)).toContain(code)
  } finally {
    await harness.close()
  }
})
