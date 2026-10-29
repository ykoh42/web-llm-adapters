import { stringifyJSON } from './json.js'
import type { JSONValue } from './types.js'

export interface ErrorInfo {
  name: string
  message: string
  causes?: { name: string; message: string }[]
  /** The envelope also preserves an explicitly undefined cause through JSON. */
  causeValue?: { value?: JSONValue | undefined }
}

type ErrorType = new (message?: string, options?: ErrorOptions) => Error

const errorTypes = new Map<string, ErrorType>([
  ['TypeError', TypeError],
  ['RangeError', RangeError],
  ['SyntaxError', SyntaxError],
  ['ReferenceError', ReferenceError],
  ['URIError', URIError],
  ['EvalError', EvalError],
])

const contextErrors = ['ContextWindowSizeExceededError', 'EmbeddingExceedContextWindowSizeError']

export function isError(error: unknown): error is Error {
  return errorType(error) === true
}

function errorType(error: unknown): boolean | undefined {
  try {
    return error instanceof Error || isCrossRealmError(error)
  } catch {
    return undefined
  }
}

function isCrossRealmError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    Object.prototype.toString.call(error) === '[object Error]'
  )
}

/** Workers may return native context errors as either strings or Errors. */
export function isContextOverflow(error: unknown): boolean {
  if (typeof error === 'string') return contextErrors.some((name) => error.startsWith(name + ':'))
  const name = errorField(error, 'name')
  if (typeof name === 'string' && contextErrors.includes(name)) return true
  const message = errorField(error, 'message')
  return typeof message === 'string' && contextErrors.some((name) => message.startsWith(name + ':'))
}

export function serializeError(error: unknown): ErrorInfo {
  const info: ErrorInfo = { name: 'Error', message: errorText(error) }
  const causes: NonNullable<ErrorInfo['causes']> = []
  const type = errorType(error)
  try {
    if (type === true) {
      const nativeError = error as Error
      const name = nativeError.name
      if (typeof name === 'string') info.name = name
      readCauses(nativeError, info, causes)
    } else if (type === undefined || typeof errorField(error, 'message') === 'string') {
      readErrorFields(error, info)
    }
  } catch {
    // Proxies may reject prototype or error-field inspection.
    readErrorFields(error, info)
  }
  if (causes.length) info.causes = causes
  return info
}

function readCauses(error: Error, info: ErrorInfo, causes: NonNullable<ErrorInfo['causes']>): void {
  let parent = error
  let cause = parent.cause
  let causeType = errorType(cause)
  while (causeType === true && causes.length < 5 && isError(cause)) {
    const name = cause.name
    causes.push({ name: typeof name === 'string' ? name : 'Error', message: errorText(cause) })
    parent = cause
    cause = parent.cause
    causeType = errorType(cause)
  }
  if (causeType === undefined) {
    const name = errorField(cause, 'name')
    const message = errorField(cause, 'message')
    if (typeof name === 'string' && typeof message === 'string') causes.push({ name, message })
  }
  if (causeType === false && 'cause' in parent) {
    try {
      info.causeValue = cause === undefined ? {} : { value: JSON.parse(stringifyJSON(cause)) }
    } catch {
      // Keep the error chain when a non-Error cause cannot be serialized.
    }
  }
}

export function errorText(error: unknown): string {
  try {
    const type = errorType(error)
    if (type === true) return String((error as Error).message)
    const message = errorField(error, 'message')
    if (typeof message === 'string') return message
    return String(error)
  } catch {
    const message = errorField(error, 'message')
    return typeof message === 'string' ? message : 'The request failed with an unreadable error.'
  }
}

function readErrorFields(error: unknown, info: ErrorInfo): void {
  const name = errorField(error, 'name')
  if (typeof name === 'string') info.name = name
  const message = errorField(error, 'message')
  if (typeof message === 'string') info.message = message
}

function errorField(error: unknown, key: string): unknown {
  if ((typeof error !== 'object' || error === null) && typeof error !== 'function') return
  try {
    return Reflect.get(error, key)
  } catch {
    return
  }
}

export function restoreError(error: ErrorInfo): Error {
  let options: ErrorOptions | undefined = error.causeValue
    ? { cause: error.causeValue.value }
    : undefined
  for (const item of [...(error.causes ?? [])].reverse())
    options = { cause: makeError(item.name, item.message, options) }
  return makeError(error.name, error.message, options)
}

function makeError(name: string, message: string, options?: ErrorOptions): Error {
  const ErrorClass = errorTypes.get(name) ?? Error
  return Object.assign(new ErrorClass(message, options), { name })
}
