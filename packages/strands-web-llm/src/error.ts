import { ContextWindowOverflowError } from '@strands-agents/sdk'

/** Workers send WebLLM errors as strings; keep context recovery available in either engine. */
export function modelError(error: unknown): unknown {
  const name = error instanceof Error ? error.name : ''
  if (
    name === 'ContextWindowSizeExceededError' ||
    (typeof error === 'string' && error.startsWith('ContextWindowSizeExceededError:'))
  ) {
    return new ContextWindowOverflowError(error instanceof Error ? error.message : String(error))
  }
  return error
}
