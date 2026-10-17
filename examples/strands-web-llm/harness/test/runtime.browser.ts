import { readFile } from 'node:fs/promises'
import { Message, type StreamOptions } from '@strands-agents/sdk'
import { test, expect } from '../../fixtures/browser.js'
import { openHarness } from './setup.js'

test('the recorded automatic archive call uses its own parameters', async ({ page }) => {
  const input: {
    messages: ReturnType<Message['toJSON']>[]
    options: StreamOptions
  } = JSON.parse(await readFile(new URL('./fixtures/archive.json', import.meta.url), 'utf8'))
  const harness = await openHarness(page)
  try {
    const stream = harness.model.streamAggregated(
      input.messages.map(Message.fromJSON),
      input.options,
    )
    for (;;) {
      const step = await stream.next()
      if (!step.done) continue
      expect(step.value.stopReason).toBe('toolUse')
      const call = step.value.message.content.find((block) => block.type === 'toolUseBlock')
      expect(call?.name).toBe('archive')
      expect([{}, { _background_execution: false }]).toContainEqual(call?.input)
      expect(step.value.message.content).toContainEqual(
        expect.objectContaining({
          type: 'reasoningBlock',
          text: expect.stringContaining('archive'),
        }),
      )
      break
    }
  } finally {
    await harness.close()
  }
})
