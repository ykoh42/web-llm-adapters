import { runInNewContext } from 'node:vm'
import { describe, expect, it } from 'vitest'
import { isFloat32Array, isUint8Array } from '../../../shared/web-llm/typed-array.js'

describe('typed array checks', () => {
  it('recognizes Uint8Array values from another realm', () => {
    expect(isUint8Array(runInNewContext('new Uint8Array([1, 2, 3])'))).toBe(true)
  })

  it('recognizes Float32Array values from another realm', () => {
    expect(isFloat32Array(runInNewContext('new Float32Array([1, 2, 3])'))).toBe(true)
  })

  it.each([new DataView(new ArrayBuffer(1)), new Uint8ClampedArray([1]), new Float64Array([1])])(
    'rejects %s as a supported media array',
    (value) => {
      expect(isUint8Array(value)).toBe(false)
      expect(isFloat32Array(value)).toBe(false)
    },
  )
})
