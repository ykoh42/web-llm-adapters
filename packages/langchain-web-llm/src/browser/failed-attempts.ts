import { applyBrowserError, serializeBrowserError, type BrowserError } from './errors.js'

export interface FailedAttempt {
  failedAttempt: BrowserError
  done: false
  native: true
}

/** Pause the browser's retry handler until the caller has handled its error. */
export function failedAttempts<T>(signal: AbortSignal) {
  let pending: Promise<T> | undefined
  let ready = false
  let failure: BrowserError | undefined
  let resume: ((error: BrowserError) => void) | undefined
  let wake: (() => void) | undefined
  function settle() {
    ready = true
    wake?.()
  }
  return {
    async handle(error: Error) {
      signal.throwIfAborted()
      try {
        await waitForCallback(
          new Promise<void>((resolve) => {
            resume = (changed) => {
              applyBrowserError(error, changed)
              resolve()
            }
            failure = serializeBrowserError(error)
            wake?.()
          }),
          signal,
        )
        signal.throwIfAborted()
      } finally {
        resume = undefined
      }
    },
    retry(error: BrowserError) {
      if (!signal.aborted) resume?.(error)
    },
    async next(read: () => Promise<T>): Promise<T | FailedAttempt> {
      const changed = new Promise<void>((resolve) => {
        wake = resolve
      })
      try {
        if (!pending) {
          pending = read()
          void pending.then(settle, settle)
        }
        if (!ready && !failure) await changed
        if (failure) {
          const error = failure
          failure = undefined
          return { failedAttempt: error, done: false, native: true }
        }
        const result = await pending!
        pending = undefined
        ready = false
        return result
      } finally {
        wake = undefined
      }
    },
  }
}

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
