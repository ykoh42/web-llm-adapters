export interface ErrorInfo {
  name: string
  message: string
  causes?: { name: string; message: string }[]
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
  if (!(error instanceof Error)) return false
  return (
    contextErrors.includes(error.name) ||
    contextErrors.some((name) => error.message.startsWith(name + ':'))
  )
}

export function serializeError(error: unknown): ErrorInfo {
  if (!(error instanceof Error)) return { name: 'Error', message: String(error) }
  const causes: NonNullable<ErrorInfo['causes']> = []
  let cause = error.cause
  while (cause instanceof Error && causes.length < 5) {
    causes.push({ name: cause.name, message: cause.message })
    cause = cause.cause
  }
  return {
    name: error.name,
    message: error.message,
    ...(causes.length && { causes }),
  }
}

export function restoreError(error: ErrorInfo): Error {
  let cause: Error | undefined
  for (const item of [...(error.causes ?? [])].reverse())
    cause = makeError(item.name, item.message, cause)
  return makeError(error.name, error.message, cause)
}

function makeError(name: string, message: string, cause?: Error): Error {
  const ErrorClass = errorTypes.get(name) ?? Error
  return Object.assign(new ErrorClass(message, cause ? { cause } : undefined), { name })
}
