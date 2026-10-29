import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { takeTurn } from './queue.js'
import { waitForCallback } from '../callback.js'
import { trackCleanup } from './cleanup.js'

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
    .then(() => {
      signal?.throwIfAborted()
      return open()
    })
    .then((chunks) => chunks[Symbol.asyncIterator]())
  let closing: Promise<void> | undefined
  let reading: Promise<IteratorResult<T>> | undefined
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
        if (reading) results.push(...(await Promise.allSettled([reading])))
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
    const cleanup = close(true)
    if (signal) trackCleanup(signal, cleanup)
    void cleanup.catch(() => {})
  }
  signal?.addEventListener('abort', cancel, { once: true })
  if (signal?.aborted) cancel()
  try {
    const iterator = await waitForCallback(source, signal)
    signal?.throwIfAborted()
    async function* chunks() {
      while (true) {
        reading = iterator.next()
        const result = signal ? await waitForCallback(reading, signal) : await reading
        reading = undefined
        if (result.done) {
          await close(false)
          return
        }
        yield result.value
      }
    }
    return { chunks: chunks(), close }
  } catch (error) {
    if (signal?.aborted) void close(false).catch(() => {})
    else await close(false).catch(() => {})
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
  let failed = false
  try {
    yield* read(call.chunks)
    done = true
  } catch (error) {
    failed = true
    throw error
  } finally {
    if (signal?.aborted) void call.close(true).catch(() => {})
    else if (failed) await call.close(true).catch(() => {})
    else await call.close(!done)
  }
}
