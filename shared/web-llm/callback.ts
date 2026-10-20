/** Cancellation must not wait for an application callback to finish. */
export async function waitForCallback<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending
  let cancel!: () => void
  try {
    return await Promise.race([
      pending,
      new Promise<never>((_, reject) => {
        cancel = () => reject(signal.reason)
        signal.addEventListener('abort', cancel, { once: true })
        if (signal.aborted) cancel()
      }),
    ])
  } finally {
    signal.removeEventListener('abort', cancel)
  }
}
