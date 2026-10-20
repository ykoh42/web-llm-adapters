import { ContextOverflowError } from '@langchain/core/errors'
import { isContextOverflow } from '../../../shared/web-llm/errors.js'

/** Preserve native failures while exposing LangChain's standard overflow marker. */
export function modelError(error: unknown): unknown {
  if (ContextOverflowError.isInstance(error)) return error
  if (error instanceof Error && error.name === 'ContextOverflowError') {
    const restored = new ContextOverflowError(error.message)
    if (error.cause instanceof Error) restored.cause = error.cause
    return restored
  }
  if (!isContextOverflow(error)) return error
  return ContextOverflowError.fromError(error instanceof Error ? error : new Error(String(error)))
}
