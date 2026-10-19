import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { takeTurn } from './queue.js'

/** Own the native stream until it closes, even if a caller abandons its iterator on abort. */
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
        // Start both operations together: an interrupt may need the iterator to close.
        const results = await Promise.allSettled([
          (async () => {
            if (interrupt) await engine.interruptGenerate()
          })(),
          source.then((iterator) => iterator.return?.()),
        ])
        for (const result of results) {
          if (result.status === 'rejected') throw result.reason
        }
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
      while (true) {
        const result = await iterator.next()
        if (result.done) {
          await close(false)
          return
        }
        yield result.value
      }
    }
    return { chunks: chunks(), close }
  } catch (error) {
    await close(false).catch(() => {})
    throw error
  }
}

/** Every protocol shares the same native stream ownership and cleanup. */
export async function* streamCall<T, R>(
  engine: MLCEngineInterface,
  open: () => Promise<AsyncIterable<T>>,
  read: (chunks: AsyncIterable<T>) => AsyncIterable<R>,
  signal?: AbortSignal,
): AsyncGenerator<R> {
  const call = await startCall(engine, open, signal)
  let done = false
  try {
    yield* read(call.chunks)
    done = true
  } finally {
    await call.close(!done)
  }
}
