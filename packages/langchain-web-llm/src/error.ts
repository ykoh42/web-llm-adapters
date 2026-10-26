import { ContextOverflowError } from '@langchain/core/errors'
import {
  errorText,
  isError,
  isContextOverflow,
  restoreError,
  serializeError,
} from '../../../shared/web-llm/errors.js'

/** Preserve native failures while exposing LangChain's standard overflow marker. */
export function modelError(error: unknown): Error {
  try {
    if (ContextOverflowError.isInstance(error)) return error
  } catch {
    // Proxies may reject prototype checks.
  }
  const nativeError = isError(error) ? error : undefined
  if (nativeError) {
    try {
      const { name, message } = nativeError
      if (name === 'ContextOverflowError') {
        const restored = new ContextOverflowError(message)
        if (isError(nativeError.cause)) restored.cause = nativeError.cause
        return restored
      }
    } catch {
      return restoreFailure(error)
    }
  }
  if (!isContextOverflow(error)) return nativeError ?? restoreFailure(error)
  return Object.assign(new ContextOverflowError(errorText(error)), { cause: error })
}

function restoreFailure(error: unknown): Error {
  const restored = restoreError(serializeError(error))
  restored.cause = error
  return restored
}
