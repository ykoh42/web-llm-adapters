/** Receives each native chunk; returned promises are awaited in order. */
export type ChunkCallback<T> = (chunk: T) => void

type RuntimeChunkCallback<T> = (chunk: T) => unknown

const nonMutatingCallbacks = new WeakSet<object>()

/** Mark an internal observer that stores chunks without changing them. */
export function markChunkCallbackNonMutating<T>(callback: ChunkCallback<T>): ChunkCallback<T> {
  nonMutatingCallbacks.add(callback)
  return callback
}

export function chunkCallbackMayMutate<T>(callback: ChunkCallback<T> | undefined): boolean {
  if (!callback) return false
  return !nonMutatingCallbacks.has(callback)
}

/** Cancellation must not wait for an application callback to finish. */
export async function waitForCallback<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending
  let cancel!: () => void
  try {
    return await Promise.race([
      new Promise<never>((_, reject) => {
        cancel = () => reject(signal.reason)
        signal.addEventListener('abort', cancel, { once: true })
        if (signal.aborted) cancel()
      }),
      pending,
    ])
  } finally {
    signal.removeEventListener('abort', cancel)
  }
}

export function callChunkCallback<T>(
  callback: RuntimeChunkCallback<T> | undefined,
  chunk: T,
  signal?: AbortSignal,
): Promise<unknown> | undefined {
  const result = callback?.(chunk)
  if (
    !result ||
    (typeof result !== 'object' && typeof result !== 'function') ||
    !('then' in result)
  )
    return
  return waitForCallback(Promise.resolve(result), signal)
}
