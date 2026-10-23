export type BrowserPacket<T, C> = (
  IteratorResult<T, undefined> | { native: true; done: false; value?: never }
) & { chunks?: C[] }

/** Native callbacks can run while a protocol parser is still waiting for a complete call. */
export function browserStream<T, C>(chunks: C[]) {
  let pending: Promise<IteratorResult<T, undefined>> | undefined
  let ready = false
  let notified = false
  let wake: (() => void) | undefined
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
        if (!ready) return { native: true, done: false, chunks: chunks.splice(0) }
        const result = await pending
        pending = undefined
        ready = false
        return { ...result, ...(chunks.length && { chunks: chunks.splice(0) }) }
      } finally {
        wake = undefined
      }
    },
  }
}
