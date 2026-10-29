/** Stop an incomplete call on cancellation or early iterator exit. */
export function watchBrowserCall(
  id: string,
  stop: (id: string) => Promise<void>,
  signal?: AbortSignal,
): (complete: boolean, failed: boolean) => Promise<void> | undefined {
  let closing: Promise<void> | undefined
  const close = () => (closing ??= stop(id))
  const cancel = () => {
    void close().catch(() => {})
  }
  signal?.addEventListener('abort', cancel, { once: true })

  return (complete, failed) => {
    signal?.removeEventListener('abort', cancel)
    if (complete) return
    if (signal?.aborted) {
      void close().catch(() => {})
      return
    }
    return failed ? close().catch(() => {}) : close()
  }
}
