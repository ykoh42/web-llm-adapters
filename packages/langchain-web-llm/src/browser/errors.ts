import { getRetryable, stampRetryable } from '@langchain/core/errors'
import {
  isError,
  restoreError,
  serializeError,
  type ErrorInfo,
} from '../../../../shared/web-llm/errors.js'
import { modelError } from '../error.js'

export interface BrowserError extends ErrorInfo {
  fields?: Record<string, unknown>
  retryable?: boolean
}

const reserved = new Set(['name', 'message', 'stack', 'cause'])

/** Keep serializable provider fields without reading own getters or losing the failure. */
export function serializeBrowserError(error: unknown): BrowserError {
  const info = serializeError(error)
  try {
    if (!(error instanceof Error) && !isError(error)) return info
  } catch {
    // A proxy may refuse prototype inspection while exposing Error fields.
  }
  const fields = Object.create(null) as Record<string, unknown>
  try {
    for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(error))) {
      if (reserved.has(key) || !Object.hasOwn(descriptor, 'value')) continue
      try {
        const value = JSON.stringify(descriptor.value)
        if (value !== undefined) fields[key] = JSON.parse(value)
      } catch {
        // A provider field may contain cycles, BigInt, or a throwing toJSON method.
      }
    }
  } catch {
    // A proxy may refuse property inspection; keep the original failure and retry marker.
  }
  let retryable: boolean | undefined
  try {
    retryable = getRetryable(error)
  } catch {
    // A proxy may refuse the retry marker lookup.
  }
  return {
    ...info,
    ...(Object.keys(fields).length && { fields }),
    ...(retryable !== undefined && { retryable }),
  }
}

export function restoreBrowserError(snapshot: BrowserError): Error {
  const error = modelError(restoreError(snapshot))
  applyBrowserError(error, snapshot)
  return error
}

/** Apply callback edits to the original error without replacing its prototype. */
export function applyBrowserError(error: Error, snapshot: BrowserError): void {
  const fields = snapshot.fields ?? {}
  const previous = serializeBrowserError(error)
  for (const key of Object.keys(previous.fields ?? {})) {
    if (Object.hasOwn(fields, key)) continue
    try {
      Reflect.deleteProperty(error, key)
    } catch {
      // Read-only provider errors must still report their original failure.
    }
  }
  for (const [key, value] of Object.entries(fields)) {
    if (!reserved.has(key)) setField(error, key, value)
  }
  setField(error, 'name', snapshot.name)
  setField(error, 'message', snapshot.message)
  applyCause(error, previous, snapshot)
  if (snapshot.retryable !== undefined) stampRetryable(error, snapshot.retryable)
}

function applyCause(error: Error, previous: BrowserError, snapshot: BrowserError): void {
  if (
    JSON.stringify(previous.causes) === JSON.stringify(snapshot.causes) &&
    JSON.stringify(previous.causeValue) === JSON.stringify(snapshot.causeValue)
  )
    return
  if (snapshot.causes?.length || snapshot.causeValue) {
    setField(error, 'cause', restoreError(snapshot).cause)
    return
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(error, 'cause')
    if (descriptor && Object.hasOwn(descriptor, 'value')) Reflect.deleteProperty(error, 'cause')
  } catch {
    // A provider error may refuse property inspection.
  }
}

function setField(error: Error, key: string, value: unknown): void {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(error, key)
    if (descriptor && !Object.hasOwn(descriptor, 'value')) return
    Object.defineProperty(error, key, {
      configurable: true,
      enumerable: true,
      writable: true,
      ...descriptor,
      value,
    })
  } catch {
    // A protected provider field cannot accept callback edits.
  }
}
