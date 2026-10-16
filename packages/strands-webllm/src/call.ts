import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { takeTurn } from './queue.js'
import { modelError } from './error.js'

/** Own the native stream until it closes, even if Strands abandons its iterator on abort. */
export async function startCall<T>(
  engine: MLCEngineInterface,
  open: () => Promise<AsyncIterable<T>>,
  signal?: AbortSignal,
) {
  const release = await takeTurn(engine, signal)
  if (signal?.aborted) {
    release()
    signal.throwIfAborted()
  }

  const source = Promise.resolve()
    .then(open)
    .then((chunks) => chunks[Symbol.asyncIterator]())
  let closing: Promise<void> | undefined
  function close(interrupt: boolean) {
    closing ??= (async () => {
      try {
        if (interrupt) engine.interruptGenerate()
        const iterator = await source
        await iterator.return?.()
      } finally {
        signal?.removeEventListener('abort', cancel)
        release()
      }
    })()
    return closing
  }
  function cancel() {
    // close() keeps the rejection for a consuming caller. An abandoned iterator has no caller.
    void close(true).catch(() => {})
  }
  signal?.addEventListener('abort', cancel, { once: true })
  if (signal?.aborted) cancel()
  try {
    const iterator = await source
    signal?.throwIfAborted()
    async function* chunks() {
      try {
        while (true) {
          const result = await iterator.next()
          if (result.done) {
            await close(false)
            return
          }
          yield result.value
        }
      } catch (error) {
        throw modelError(error)
      }
    }
    return { chunks: chunks(), close }
  } catch (error) {
    await close(false).catch(() => {})
    throw modelError(error)
  }
}
