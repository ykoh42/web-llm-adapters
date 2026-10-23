import { ContextOverflowError } from '@langchain/core/errors'
import {
  errorText,
  isContextOverflow,
  restoreError,
  serializeError,
} from '../../../shared/web-llm/errors.js'

/** Preserve native failures while exposing LangChain's standard overflow marker. */
export function modelError(error: unknown): unknown {
  if (ContextOverflowError.isInstance(error)) return error
  if (error instanceof Error) {
    try {
      const { name, message } = error
      if (name === 'ContextOverflowError') {
        const restored = new ContextOverflowError(message)
        if (error.cause instanceof Error) restored.cause = error.cause
        return restored
      }
    } catch {
      const restored = restoreError(serializeError(error))
      restored.cause = error
      return restored
    }
  }
  if (!isContextOverflow(error))
    return error instanceof Error ? error : new Error(errorText(error), { cause: error })
  return ContextOverflowError.fromError(error instanceof Error ? error : new Error(String(error)))
}
