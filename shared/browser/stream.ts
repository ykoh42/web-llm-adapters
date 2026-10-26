export type BrowserPacket<T, C> = (
  IteratorResult<T, undefined> | { native: true; done: false; value?: never }
) & { chunks?: C[] }

/** Native callbacks can run while a protocol parser is still waiting for a complete call. */
export function browserStream<T, C>(initialChunks: C[] = []) {
  let chunks = initialChunks
  let pending: Promise<IteratorResult<T, undefined>> | undefined
  let ready = false
  let notified = false
  let wake: (() => void) | undefined
  const takeChunks = () => {
    const batch = chunks
    chunks = []
    return batch
  }
  function settle() {
    ready = true
    wake?.()
  }
  return {
    push(chunk: C) {
      chunks.push(chunk)
      const notify = wake
      if (!notify || notified) return
      notified = true
      // Let parser microtasks finish before sending a separate native packet.
      if (typeof globalThis.scheduler?.yield === 'function')
        void scheduler.yield().then(notify, notify)
      else notify()
    },
    takeChunks,
    async next(stream: AsyncGenerator<T>): Promise<BrowserPacket<T, C>> {
      notified = false
      const changed = new Promise<void>((resolve) => {
        wake = resolve
      })
      try {
        if (!pending) {
          pending = stream.next()
          void pending.then(settle, settle)
        }
        if (!ready && !chunks.length) await changed
        if (!ready) return { native: true, done: false, chunks: takeChunks() }
        const result = await pending
        pending = undefined
        ready = false
        return { ...result, ...(chunks.length && { chunks: takeChunks() }) }
      } finally {
        wake = undefined
      }
    },
  }
}
