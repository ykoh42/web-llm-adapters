import { AsyncCaller, type AsyncCallerParams } from '@langchain/core/utils/async_caller'
import { waitForCallback } from '../../../../shared/web-llm/callback.js'

export class SignalCaller extends AsyncCaller {
  constructor(params: AsyncCallerParams, signal: AbortSignal) {
    super(params)
    const onFailedAttempt = this.onFailedAttempt
    this.onFailedAttempt = (error: unknown) => {
      signal.throwIfAborted()
      return waitForCallback(
        Promise.resolve().then(() => {
          signal.throwIfAborted()
          return onFailedAttempt?.(error)
        }),
        signal,
      )
    }
  }
}
