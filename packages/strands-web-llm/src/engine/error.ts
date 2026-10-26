import { errorText, isContextOverflow } from '../../../../shared/web-llm/errors.js'
import { ModelError as ProtocolError } from '../../../../shared/web-llm/types.js'
import { ModelError, ContextWindowOverflowError } from '@strands-agents/sdk'

/** Workers send WebLLM errors as strings; keep context recovery available in either engine. */
export function modelError(error: unknown): unknown {
  if (isContextWindowOverflowError(error)) return error
  if (isContextOverflow(error)) {
    return Object.assign(new ContextWindowOverflowError(errorText(error)), { cause: error })
  }
  if (isProtocolError(error)) return new ModelError(error.message, { cause: error })
  return error
}

function isContextWindowOverflowError(error: unknown): error is ContextWindowOverflowError {
  try {
    return error instanceof ContextWindowOverflowError
  } catch {
    return false
  }
}

function isProtocolError(error: unknown): error is ProtocolError {
  try {
    return error instanceof ProtocolError
  } catch {
    return false
  }
}
