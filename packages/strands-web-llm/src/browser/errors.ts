import { ContextWindowOverflowError, ModelError, ModelThrottledError } from '@strands-agents/sdk'
import { restoreError } from '../../../../shared/web-llm/errors.js'
import type { BrowserError } from './types.js'

export { serializeError as toBrowserError } from '../../../../shared/web-llm/errors.js'

export function fromBrowserError(error: BrowserError): Error {
  const restored = restoreError(error)
  const options = Object.hasOwn(restored, 'cause') ? { cause: restored.cause } : undefined
  if (error.name === 'ContextWindowOverflowError') {
    const overflow = new ContextWindowOverflowError(error.message)
    if (options) Object.assign(overflow, options)
    return overflow
  }
  if (error.name === 'ModelError') return new ModelError(error.message, options)
  if (error.name === 'ModelThrottledError') return new ModelThrottledError(error.message, options)
  return restored
}
