import { applyBrowserError, serializeBrowserError, type BrowserError } from './errors.js'
import { waitForCallback } from '../../../../shared/web-llm/callback.js'

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
