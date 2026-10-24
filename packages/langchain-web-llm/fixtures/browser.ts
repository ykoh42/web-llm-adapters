import { vi } from 'vitest'
import { BrowserChatWebLLM, createBrowserBridge, type BrowserPage } from '../src/index.js'
import { chunk, setup } from './engine.js'

export function browser(...items: ReturnType<typeof chunk>[]) {
  vi.stubGlobal('scheduler', { yield: () => new Promise<void>((resolve) => setImmediate(resolve)) })
  const result = setup(...items)
  const bridge = createBrowserBridge(result.engine)
  vi.stubGlobal('window', { webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  return {
    ...result,
    bridge,
    page,
    model: new BrowserChatWebLLM(page, { modelId: 'Qwen3-8B-q4f16_1-MLC' }),
  }
}
