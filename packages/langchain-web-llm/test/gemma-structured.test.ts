import { afterEach, expect, it, vi } from 'vitest'
import { createAgent } from 'langchain'
import { z } from 'zod'
import type { Completion } from '@mlc-ai/web-llm'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
afterEach(() => vi.unstubAllGlobals())

function setupModels(browser: boolean) {
  const { engine, create } = setup(chunk({ content: '{"answer":42}' }, 'stop'))
  const complete = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
    (async function* () {
      const response: Completion = {
        id: 'native',
        model: modelId,
        object: 'text_completion',
        created: 0,
        choices: [{ index: 0, text: 'answer:42}<tool_call|>', finish_reason: 'stop' }],
      }
      yield response
    })(),
  )
  const bridge = createBrowserBridge(engine)
  vi.stubGlobal('window', { webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const result = await run(structuredClone(arg))
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  return {
    create,
    complete,
    bridge,
    model(options: { thinking?: boolean; profile?: { structuredOutput: boolean } }) {
      const config = {
        modelId,
        ...(options.profile && { profile: options.profile }),
        request:
          options.thinking === undefined
            ? {}
            : { extra_body: { enable_thinking: options.thinking } },
      }
      return browser ? new BrowserChatWebLLM(page, config) : new ChatWebLLM({ engine, ...config })
    },
  }
}

for (const browser of [false, true]) {
  it.each([false, true])(
    `lets the agent choose tool output for Gemma thinking (browser: ${browser}, profile override: %s)`,
    async (override) => {
      const state = setupModels(browser)
      const profile = override ? { structuredOutput: true } : undefined
      try {
        const model = state.model({ thinking: true, ...(profile && { profile }) })
        const agent = createAgent({ model, responseFormat: z.object({ answer: z.number() }) })
        const result = await agent.invoke({
          messages: [{ role: 'user', content: 'Six times seven?' }],
        })
        expect(result.structuredResponse).toEqual({ answer: 42 })
        expect(model.profile.structuredOutput).toBe(false)
        expect(model.profile.reasoningOutput).toBe(true)
        expect(profile?.structuredOutput).toBe(override ? true : undefined)
        expect(state.complete).toHaveBeenCalledOnce()
        expect(state.create).not.toHaveBeenCalled()
        expect(state.complete.mock.calls[0]?.[0]).not.toHaveProperty('response_format')
        expect(state.complete.mock.calls[0]?.[0].prompt).toContain('<|think|>')
      } finally {
        await state.bridge.close()
      }
    },
  )
  it.each([false, undefined])(
    `keeps native Gemma JSON when thinking is %s (browser: ${browser})`,
    async (thinking) => {
      const state = setupModels(browser)
      try {
        const model = state.model(thinking === undefined ? {} : { thinking })
        const agent = createAgent({ model, responseFormat: z.object({ answer: z.number() }) })
        const result = await agent.invoke({
          messages: [{ role: 'user', content: 'Six times seven?' }],
        })
        expect(result.structuredResponse).toEqual({ answer: 42 })
        expect(model.profile.structuredOutput).toBe(true)
        expect(state.create).toHaveBeenCalledOnce()
        expect(state.create.mock.calls[0]?.[0].response_format?.schema).toContain('answer')
        expect(state.complete).not.toHaveBeenCalled()
      } finally {
        await state.bridge.close()
      }
    },
  )
}
