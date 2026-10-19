import { ContextWindowOverflowError, ModelError, ModelThrottledError } from '@strands-agents/sdk'
import type { BrowserError } from './types.js'

export function toBrowserError(error: unknown): BrowserError {
  if (!(error instanceof Error)) return { name: 'Error', message: String(error) }
  return {
    name: error.name,
    message: error.message,
    ...(error.cause instanceof Error && { causes: causes(error.cause) }),
  }
}

export function fromBrowserError(error: BrowserError): Error {
  const cause = restoreCauses(error.causes)
  const options = cause ? { cause } : undefined
  if (error.name === 'ContextWindowOverflowError') {
    const restored = new ContextWindowOverflowError(error.message)
    if (cause) Object.assign(restored, { cause })
    return restored
  }
  if (error.name === 'ModelError') return new ModelError(error.message, options)
  if (error.name === 'ModelThrottledError') return new ModelThrottledError(error.message, options)
  return Object.assign(new Error(error.message, options), { name: error.name })
}

function causes(error: Error, depth = 0): { name: string; message: string }[] {
  const result = [{ name: error.name, message: error.message }]
  if (depth >= 4 || !(error.cause instanceof Error)) return result
  return [...result, ...causes(error.cause, depth + 1)]
}

function restoreCauses(causes: BrowserError['causes']): Error | undefined {
  let cause: Error | undefined
  for (const item of [...(causes ?? [])].reverse()) {
    cause = Object.assign(new Error(item.message, cause ? { cause } : undefined), {
      name: item.name,
    })
  }
  return cause
}
