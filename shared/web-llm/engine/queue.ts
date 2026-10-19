import type { MLCEngineInterface } from '@mlc-ai/web-llm'

const key = Symbol.for('web-llm-adapters.engine-queue.v1')
const registry = globalThis as typeof globalThis & {
  [key]?: WeakMap<MLCEngineInterface, Promise<void>>
}
// Bundled adapters still share one queue when they use the same engine.
const tails = (registry[key] ??= new WeakMap<MLCEngineInterface, Promise<void>>())

/** Reserve one engine turn. A cancelled waiter keeps its place until earlier work closes. */
export async function takeTurn(engine: MLCEngineInterface, signal?: AbortSignal) {
  signal?.throwIfAborted()
  const previous = tails.get(engine) ?? Promise.resolve()
  let release!: () => void
  const finished = new Promise<void>((resolve) => {
    release = resolve
  })
  const tail = previous.then(() => finished)
  tails.set(engine, tail)
  void tail.then(() => {
    if (tails.get(engine) === tail) tails.delete(engine)
  })
  try {
    await new Promise<void>((resolve, reject) => {
      function cancel() {
        signal?.removeEventListener('abort', cancel)
        reject(signal?.reason)
      }
      signal?.addEventListener('abort', cancel, { once: true })
      void previous.then(() => {
        signal?.removeEventListener('abort', cancel)
        resolve()
      })
      if (signal?.aborted) cancel()
    })
    signal?.throwIfAborted()
    return release
  } catch (error) {
    release()
    throw error
  }
}
