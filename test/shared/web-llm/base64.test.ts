import { runInNewContext } from 'node:vm'
import { Buffer } from 'node:buffer'
import { describe, expect, it, vi } from 'vitest'
import { base64Data } from '../../../shared/web-llm/base64.js'

describe('base64Data', () => {
  it('encodes Uint8Array values from another realm', () => {
    const bytes: unknown = runInNewContext('new Uint8Array([1, 2, 3])')

    expect(base64Data(bytes)).toBe('AQID')
  })

  it('encodes views across browser fallback chunk boundaries', () => {
    const bytes = runInNewContext(
      'Uint8Array.from({ length: 24580 }, (_, index) => index % 256).subarray(3)',
    ) as Uint8Array
    const method = Object.getOwnPropertyDescriptor(Uint8Array.prototype, 'toBase64')
    vi.stubGlobal('Buffer', undefined)
    if (method)
      Object.defineProperty(Uint8Array.prototype, 'toBase64', { ...method, value: undefined })

    try {
      expect(base64Data(bytes)).toBe(Buffer.from(Array.from(bytes)).toString('base64'))
    } finally {
      if (method) Object.defineProperty(Uint8Array.prototype, 'toBase64', method)
      vi.unstubAllGlobals()
    }
  })

  it.each([new DataView(new ArrayBuffer(1)), new Uint8ClampedArray([1]), new Float32Array([1])])(
    'does not encode %s',
    (value) => {
      expect(base64Data(value)).toBeUndefined()
    },
  )
})
