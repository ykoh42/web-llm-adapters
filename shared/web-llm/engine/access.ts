import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { takeTurn } from './queue.js'

/**
 * Run native WebLLM work without overlapping adapter calls on the same engine.
 * Await all work and consume streams inside the callback. Do not nest same-engine calls.
 */
export async function withEngine<T>(
  engine: MLCEngineInterface,
  run: (engine: MLCEngineInterface) => T | Promise<T>,
): Promise<T> {
  const release = await takeTurn(engine)
  try {
    const result = await run(engine)
    if (result && typeof result === 'object' && Symbol.asyncIterator in result) {
      // A returned iterator would outlive this turn. Consume it inside the callback instead.
      const source = result as AsyncIterable<unknown>
      const results = await Promise.allSettled([
        Promise.resolve().then(() => engine.interruptGenerate()),
        Promise.resolve().then(() => source[Symbol.asyncIterator]().return?.()),
      ])
      for (const result of results) if (result.status === 'rejected') throw result.reason
      throw new TypeError('Consume native streams inside withEngine().')
    }
    return result
  } finally {
    release()
  }
}
