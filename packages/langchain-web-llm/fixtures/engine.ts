import { ChatWebLLM } from '../src/index.js'
import { setupEngine } from '../../../fixtures/web-llm.js'
import { vi } from 'vitest'

export { chunk, chunks, collect, pause } from '../../../fixtures/web-llm.js'

/** Advance retry backoff without making tests wait for network-style delays. */
export async function finishRetries<T>(run: () => Promise<T>): Promise<T> {
  vi.useFakeTimers()
  try {
    const pending = run().then(
      (value) => ({ value }),
      (error: unknown) => ({ error }),
    )
    await vi.runAllTimersAsync()
    const result = await pending
    if ('error' in result) throw result.error
    return result.value
  } finally {
    vi.useRealTimers()
  }
}

export function setup(...items: Parameters<typeof setupEngine>) {
  const { engine, create, stop } = setupEngine(...items)
  const model = new ChatWebLLM({ engine, modelId: 'test-model' })
  return { engine, model, create, stop }
}
