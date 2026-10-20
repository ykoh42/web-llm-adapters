import { isContextOverflow } from '../../../../shared/web-llm/errors.js'
import { ModelError as ProtocolError } from '../../../../shared/web-llm/types.js'
import { ModelError, ContextWindowOverflowError } from '@strands-agents/sdk'

/** Workers send WebLLM errors as strings; keep context recovery available in either engine. */
export function modelError(error: unknown): unknown {
  if (error instanceof ContextWindowOverflowError) return error
  if (isContextOverflow(error)) {
    const native = error instanceof Error ? error : new Error(String(error))
    return Object.assign(new ContextWindowOverflowError(native.message), { cause: native })
  }
  if (error instanceof ProtocolError) return new ModelError(error.message, { cause: error })
  return error
}
