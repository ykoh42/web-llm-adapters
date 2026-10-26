import { isUint8Array } from './typed-array.js'

export function base64Data(value: string | Uint8Array): string
export function base64Data(value: unknown): string | undefined
export function base64Data(value: unknown): string | undefined {
  if (typeof value === 'string') return value
  if (!isUint8Array(value)) return undefined
  if (!value.length) return ''
  const prototype: Uint8Array & { toBase64?: () => string } = Uint8Array.prototype
  if (typeof prototype.toBase64 === 'function') return prototype.toBase64.call(value)
  if (typeof globalThis.Buffer === 'function')
    return globalThis.Buffer.from(value.buffer, value.byteOffset, value.byteLength).toString(
      'base64',
    )

  const parts: string[] = []
  for (let index = 0; index < value.length; index += 0x6000)
    parts.push(btoa(String.fromCharCode(...value.subarray(index, index + 0x6000))))
  return parts.join('')
}
