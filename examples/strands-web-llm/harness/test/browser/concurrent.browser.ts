import { modelCheck } from '../../../../../scripts/release/support.js'
import { writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { Message, TextBlock } from '@strands-agents/sdk'
import { test, expect } from '../../../fixtures/browser.js'
import { openHarness, useTool } from '../../fixtures/harness.js'

test(
  'concurrent Harness agents keep tools and answers isolated on one Gemma engine',
  modelCheck('engine', 'concurrency'),
  async ({ page }) => {
    const harness = await openHarness(page)
    try {
      const codes = ['red-', 'blue-'].map((prefix) => prefix + crypto.randomUUID().slice(0, 8))
      const paths = codes.map((_, i) => join(harness.dir, `note-${i}.txt`))
      await Promise.all(paths.map((path, i) => writeFile(path, `The secret code is ${codes[i]}.`)))
      const agents = await Promise.all(codes.map(() => harness.create()))
      for (const agent of agents) useTool(agent, 'read')
      const results = await Promise.all(
        agents.map((agent, i) =>
          agent.invoke(
            `Use the read tool to read ${paths[i]}. Reply with the secret code from that file.`,
            { limits: { turns: 4 } },
          ),
        ),
      )
      for (const [i, result] of results.entries()) {
        const text = result.lastMessage.content
          .filter((b) => b.type === 'textBlock')
          .map((b) => b.text)
          .join('')
        expect(text).toContain(codes[i])
        expect(text).not.toContain(codes[1 - i])
        expect(agents[i]!.messages.flatMap((m) => m.content)).toContainEqual(
          expect.objectContaining({ type: 'toolResultBlock', status: 'success' }),
        )
        expect(
          agents[i]!.messages.flatMap((message) => message.content).some(
            (block) => block.type === 'reasoningBlock' && block.text,
          ),
        ).toBe(true)
      }
    } finally {
      await harness.close()
    }
  },
)

test(
  'cancelling and closing browser calls leaves the engine usable',
  modelCheck('engine', 'cancel'),
  async ({ page }) => {
    const harness = await openHarness(page)
    try {
      await page.evaluate(() => {
        const seen = new Set<string>()
        const next = window.webllmBridge.next
        window.webllmBridge.next = (id) => {
          const result = next(id)
          seen.add(id)
          document.documentElement.dataset.harnessCalls = String(seen.size)
          return result
        }
      })
      const messages = (text: string) => [
        new Message({ role: 'user', content: [new TextBlock(text)] }),
      ]
      const active = harness.auxModel.stream(
        messages('Explain why the sky is blue in three sentences.'),
      )
      let text = ''
      while (!text) {
        const result = await active.next()
        if (result.done) throw new Error('The active model returned no text')
        const event = result.value
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          text += event.delta.text
      }
      const controller = new AbortController()
      const cancelled = harness.auxModel.stream(messages('Say cancelled.'), {
        cancelSignal: controller.signal,
      })
      const waiting = cancelled.next().then(
        () => undefined,
        (error) => error,
      )
      await page.waitForFunction(() => document.documentElement.dataset.harnessCalls === '2')
      controller.abort(new Error('Cancel only the waiter'))
      expect(await waiting).toMatchObject({ message: 'Cancel only the waiter' })
      await cancelled.return(undefined)
      for await (const event of active) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          text += event.delta.text
      }
      expect(text.toLowerCase()).toContain('blue')
      let later = ''
      for await (const event of harness.auxModel.stream(messages('Reply with only pineapple.'))) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          later += event.delta.text
      }
      expect(later.toLowerCase()).toMatch(/pineapple|🍍/)

      const engine = await page.evaluateHandle(() => window.webllmBridge.withEngine((e) => e))
      const unfinished = harness.auxModel.stream(
        messages('Explain the history of mathematics in detail.'),
      )
      while (true) {
        const result = await unfinished.next()
        if (result.done) throw new Error('The model returned no text')
        if (
          result.value.type === 'modelContentBlockDeltaEvent' &&
          result.value.delta.type === 'textDelta'
        )
          break
      }
      await page.evaluate(() => window.webllmBridge.close())
      await unfinished.return(undefined)
      await expect(harness.auxModel.stream(messages('Hello')).next()).rejects.toThrow('closed')
      const answer = await engine.evaluate(async (engine) => {
        const result = await engine.chat.completions.create({
          model: 'gemma-4-E2B-it-q4f16_1-MLC',
          messages: [{ role: 'user', content: 'What is 1 + 1?' }],
          temperature: 0,
          max_tokens: 128,
          extra_body: { enable_thinking: false },
        })
        return result.choices[0]?.message.content
      })
      expect(answer).toContain('2')
      await engine.dispose()
    } finally {
      await harness.close()
    }
  },
)
