import { runInNewContext } from 'node:vm'
import { describe, expect, it, vi } from 'vitest'
import { ContextOverflowError, getRetryable, stampRetryable } from '@langchain/core/errors'
import {
  applyBrowserError,
  restoreBrowserError,
  serializeBrowserError,
} from '../src/browser/errors.js'

function transfer(error: unknown) {
  return restoreBrowserError(JSON.parse(JSON.stringify(serializeBrowserError(error))))
}

describe('browser retry errors', () => {
  it('preserves native error types and causes from another realm', () => {
    const error = Object.assign(
      runInNewContext(
        'new TypeError("Native failure", { cause: new RangeError("Invalid input") })',
      ) as Error,
      { code: 'FOREIGN_ERROR' },
    )
    stampRetryable(error, true)

    const restored = transfer(error)

    expect(restored).toBeInstanceOf(TypeError)
    expect(restored.message).toBe('Native failure')
    expect(restored.cause).toBeInstanceOf(RangeError)
    expect((restored.cause as Error).message).toBe('Invalid input')
    expect(restored).toHaveProperty('code', 'FOREIGN_ERROR')
    expect(getRetryable(restored)).toBe(true)
  })

  it('preserves native error types, causes, retry hints, and provider fields', () => {
    const error = Object.assign(new TypeError('Failed to fetch', { cause: new Error('Offline') }), {
      code: 'GPU_RETRY',
      status: 429,
      retryAfterMs: 3000,
      response: { status: 429, headers: { 'retry-after': '3' } },
    })
    stampRetryable(error, true)
    const restored = transfer(error)

    expect(restored).toBeInstanceOf(TypeError)
    expect(restored).toMatchObject({
      message: 'Failed to fetch',
      cause: { message: 'Offline' },
      code: 'GPU_RETRY',
      status: 429,
      retryAfterMs: 3000,
      response: error.response,
    })
    expect(getRetryable(restored)).toBe(true)
  })

  it('restores the LangChain overflow brand and keeps provider fields', () => {
    const error = Object.assign(new Error('Too many tokens'), {
      name: 'ContextOverflowError',
      modelId: 'loaded-model',
    })
    stampRetryable(error, false)
    const restored = transfer(error)

    expect(ContextOverflowError.isInstance(restored)).toBe(true)
    expect(getRetryable(restored)).toBe(false)
    expect(restored).toHaveProperty('modelId', 'loaded-model')
  })

  it('skips unsafe fields independently without invoking getters', () => {
    const getter = vi.fn(() => {
      throw new Error('A getter must not run')
    })
    const cycle: Record<string, unknown> = {}
    cycle.self = cycle
    const error = Object.assign(new Error('Opening failed'), {
      code: 'KEEP',
      cycle,
      integer: 1n,
      callback: () => {},
      omitted: undefined,
      broken: {
        toJSON() {
          throw new Error('Cannot encode this field')
        },
      },
    })
    Object.defineProperty(error, 'getter', { get: getter, enumerable: true })

    expect(serializeBrowserError(error)).toEqual({
      name: 'Error',
      message: 'Opening failed',
      fields: { code: 'KEEP' },
    })
    expect(getter).not.toHaveBeenCalled()
  })

  it('keeps the original failure and retry marker when provider fields cannot be inspected', () => {
    const native = stampRetryable(new TypeError('Native failure'), false)
    const error = new Proxy(native, {
      ownKeys() {
        throw new Error('Cannot inspect provider fields')
      },
    })

    expect(serializeBrowserError(error)).toEqual({
      name: 'TypeError',
      message: 'Native failure',
      retryable: false,
    })
  })

  it('keeps the original failure when retryability metadata cannot be inspected', () => {
    const native = stampRetryable(new TypeError('Native failure'), false)
    const error = new Proxy(native, {
      getOwnPropertyDescriptor(target, key) {
        if (typeof key === 'symbol') throw new Error('Cannot inspect retry metadata')
        return Reflect.getOwnPropertyDescriptor(target, key)
      },
    })

    expect(serializeBrowserError(error)).toEqual({
      name: 'TypeError',
      message: 'Native failure',
    })
  })

  it('keeps the original failure when a proxy rejects prototype inspection', () => {
    const native = stampRetryable(new TypeError('Native failure'), false)
    const error = new Proxy(native, {
      getPrototypeOf() {
        throw new Error('Cannot inspect provider prototype')
      },
    })

    expect(serializeBrowserError(error)).toEqual({
      name: 'TypeError',
      message: 'Native failure',
      retryable: false,
    })
  })

  it('applies callback changes and deletions while keeping original unencodable fields', () => {
    class ProviderError extends Error {}
    const cycle: Record<string, unknown> = {}
    cycle.self = cycle
    const original = Object.assign(new ProviderError('Opening failed'), {
      code: 'OLD',
      removed: 'Delete this',
      retryAfterMs: 1000,
      cycle,
    })
    const edited = transfer(original)
    Object.assign(edited, { code: 'NEW', retryAfterMs: 4000, added: { ok: true } })
    Reflect.deleteProperty(edited, 'removed')
    edited.message = 'Try again later'
    stampRetryable(edited, true)

    applyBrowserError(original, serializeBrowserError(edited))

    expect(original).toBeInstanceOf(ProviderError)
    expect(original).toMatchObject({
      code: 'NEW',
      retryAfterMs: 4000,
      added: { ok: true },
      message: 'Try again later',
      cycle,
    })
    expect(original).not.toHaveProperty('removed')
    expect(getRetryable(original)).toBe(true)
  })

  it('keeps the original failure when its properties cannot be inspected', () => {
    const cause = new Error('Offline')
    const original = new Proxy(new Error('Native failure', { cause }), {
      getOwnPropertyDescriptor() {
        throw new Error('Cannot inspect provider fields')
      },
    })
    const edited = transfer(original)
    edited.message = 'Edited'
    Reflect.deleteProperty(edited, 'cause')

    expect(() => applyBrowserError(original, serializeBrowserError(edited))).not.toThrow()
    expect(original.message).toBe('Native failure')
    expect(original.cause).toBe(cause)
  })

  it('preserves reserved JSON keys without changing error prototypes', () => {
    const original = new Error('Opening failed')
    Object.defineProperty(original, '__proto__', {
      value: { provider: true },
      enumerable: true,
      writable: true,
      configurable: true,
    })
    const restored = transfer(original)
    expect(Object.getPrototypeOf(restored)).toBe(Error.prototype)
    expect(Object.hasOwn(restored, '__proto__')).toBe(true)
    expect(Object.getOwnPropertyDescriptor(restored, '__proto__')?.value).toEqual({
      provider: true,
    })

    applyBrowserError(original, serializeBrowserError(restored))
    expect(Object.getPrototypeOf(original)).toBe(Error.prototype)
    expect(Object.prototype).not.toHaveProperty('provider')
  })

  it('keeps unchanged native causes and applies explicit cause edits', () => {
    const cause = Object.assign(new Error('Offline'), { provider: true })
    const original = new Error('Opening failed', { cause })
    const edited = transfer(original)

    applyBrowserError(original, serializeBrowserError(edited))
    expect(original.cause).toBe(cause)

    edited.cause = new TypeError('Changed cause')
    applyBrowserError(original, serializeBrowserError(edited))
    expect(original.cause).toBeInstanceOf(TypeError)
    expect(original.cause).toHaveProperty('message', 'Changed cause')

    Reflect.deleteProperty(edited, 'cause')
    applyBrowserError(original, serializeBrowserError(edited))
    expect(original).not.toHaveProperty('cause')
  })

  it('does not replace a frozen error with a failed callback edit', () => {
    const original = Object.freeze(Object.assign(new Error('Original'), { code: 'OLD' }))
    const edited = transfer(original)
    Object.assign(edited, { code: 'NEW', message: 'Edited' })

    expect(() => applyBrowserError(original, serializeBrowserError(edited))).not.toThrow()
    expect(original).toHaveProperty('code', 'OLD')
    expect(original.message).toBe('Original')
  })

  it('keeps a read-only cause getter when its transported copy is edited', () => {
    const cause = new TypeError('Original cause')
    const original = Object.defineProperty(new Error('Native failure'), 'cause', {
      configurable: true,
      get: () => cause,
    })
    const edited = transfer(original)
    Reflect.deleteProperty(edited, 'cause')

    applyBrowserError(original, serializeBrowserError(edited))

    expect(original.cause).toBe(cause)
    expect(Object.getOwnPropertyDescriptor(original, 'cause')?.get).toBeTypeOf('function')
  })

  it('bounds cyclic cause chains to five entries', () => {
    const error = new Error('Opening failed')
    error.cause = error
    const snapshot = serializeBrowserError(error)
    expect(snapshot.causes).toHaveLength(5)
    expect(() => transfer(error)).not.toThrow()
  })

  it.each(['Worker failed', null, 0])('restores non-error failures: %s', (error) => {
    expect(transfer(error).message).toBe(String(error))
  })
})
