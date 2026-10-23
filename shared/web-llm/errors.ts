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

/** Workers may return native context errors as either strings or Errors. */
export function isContextOverflow(error: unknown): boolean {
  if (typeof error === 'string') return contextErrors.some((name) => error.startsWith(name + ':'))
  try {
    if (!(error instanceof Error)) return false
    if (contextErrors.includes(error.name)) return true
    const message = error.message
    return (
      typeof message === 'string' && contextErrors.some((name) => message.startsWith(name + ':'))
    )
  } catch {
    return false
  }
}

export function serializeError(error: unknown): ErrorInfo {
  const info: ErrorInfo = { name: 'Error', message: errorText(error) }
  const causes: NonNullable<ErrorInfo['causes']> = []
  try {
    if (error instanceof Error) {
      const name = error.name
      if (typeof name === 'string') info.name = name
      let parent = error
      let cause = parent.cause
      while (cause instanceof Error && causes.length < 5) {
        const name = cause.name
        causes.push({ name: typeof name === 'string' ? name : 'Error', message: errorText(cause) })
        parent = cause
        cause = parent.cause
      }
      if ('cause' in parent && !(cause instanceof Error))
        info.causeValue = cause === undefined ? {} : { value: JSON.parse(stringifyJSON(cause)) }
    }
  } catch {
    // Provider getters or unserializable causes must not replace the original failure.
  }
  if (causes.length) info.causes = causes
  return info
}

export function errorText(error: unknown): string {
  try {
    return String(error instanceof Error ? error.message : error)
  } catch {
    return 'The request failed with an unreadable error.'
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
