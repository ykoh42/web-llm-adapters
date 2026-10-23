import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import { BrowserModel, createBrowserBridge, type BrowserPage } from '../../src/index.js'
import { chunk, chunks, collect, setup } from '../../fixtures/engine.js'

const models = ['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']

afterEach(() => vi.unstubAllGlobals())

for (const browser of [false, true]) {
  describe(`Qwen token estimates (browser: ${browser})`, () => {
    it.each(models)('%s excludes thoughts omitted after a new question', async (modelId) => {
      const local = setup()
      local.create.mockImplementation(async () => chunks(chunk({ content: 'Done' }, 'stop')))
      local.model.updateConfig({ modelId })
      const bridge = createBrowserBridge(local.engine)
      vi.stubGlobal('window', { webllmBridge: bridge })
      const page: BrowserPage = {
        async evaluate(run, input) {
          const result = await run(structuredClone(input))
          return result === undefined ? result : JSON.parse(JSON.stringify(result))
        },
      }
      const model = browser ? new BrowserModel(page, { modelId }) : local.model
      const history = (thought: string) => [
        new Message({ role: 'user', content: [new TextBlock('First question.')] }),
        new Message({
          role: 'assistant',
          content: [new ReasoningBlock({ text: thought }), new TextBlock('Earlier answer.')],
        }),
        new Message({ role: 'user', content: [new TextBlock('Next question.')] }),
      ]
      const short = history('Earlier thought.')
      const long = history('Earlier thought. '.repeat(2000))
      const snapshot = JSON.stringify(long)
      for (const message of long) {
        for (const block of message.content) Object.freeze(block)
        Object.freeze(message.content)
        Object.freeze(message)
      }
      Object.freeze(long)
      try {
        await collect(model.stream(short))
        await collect(model.stream(long))
        expect(local.create.mock.calls[1]?.[0]).toEqual(local.create.mock.calls[0]?.[0])
        expect(JSON.stringify(local.create.mock.calls[0]?.[0])).not.toContain('Earlier thought.')
        expect(await model.countTokens(long)).toBe(await model.countTokens(short))
        expect(JSON.stringify(long)).toBe(snapshot)
      } finally {
        await bridge.close()
      }
    })

    it.each(models)('%s follows its current tool-loop thinking policy', async (modelId) => {
      const { model, engine } = setup()
      model.updateConfig({ modelId })
      const page: BrowserPage = {
        async evaluate() {
          throw new Error('Token estimates must not call the browser.')
        },
      }
      const selected = browser ? new BrowserModel(page, { modelId }) : model
      const history = (thought: string) => [
        new Message({ role: 'user', content: [new TextBlock('Look up A7.')] }),
        new Message({
          role: 'assistant',
          content: [
            new ReasoningBlock({ text: thought }),
            new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: { code: 'A7' } }),
          ],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call',
              status: 'success',
              content: [new TextBlock('42')],
            }),
          ],
        }),
      ]
      const short = await selected.countTokens(history('Current thought.'))
      const long = await selected.countTokens(history('Current thought. '.repeat(100)))
      if (modelId.startsWith('Qwen2.5-')) expect(long).toBe(short)
      else expect(long).toBeGreaterThan(short)
      expect(engine.chat.completions.create).not.toHaveBeenCalled()
    })
  })
}
