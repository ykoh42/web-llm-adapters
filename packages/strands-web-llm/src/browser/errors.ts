import { ContextWindowOverflowError, ModelError, ModelThrottledError } from '@strands-agents/sdk'
import { restoreError } from '../../../../shared/web-llm/errors.js'
import type { BrowserError } from './types.js'

export { serializeError as toBrowserError } from '../../../../shared/web-llm/errors.js'

export function fromBrowserError(error: BrowserError): Error {
  const restored = restoreError(error)
  const cause = restored.cause instanceof Error ? restored.cause : undefined
  const options = cause ? { cause } : undefined
  if (error.name === 'ContextWindowOverflowError') {
    const overflow = new ContextWindowOverflowError(error.message)
    if (cause) Object.assign(overflow, { cause })
    return overflow
  }
  if (error.name === 'ModelError') return new ModelError(error.message, options)
  if (error.name === 'ModelThrottledError') return new ModelThrottledError(error.message, options)
  return restored
}
