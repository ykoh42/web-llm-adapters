export function isUint8Array(value: unknown): value is Uint8Array {
  return isTypedArray(value, 'Uint8Array')
}

export function isFloat32Array(value: unknown): value is Float32Array {
  return isTypedArray(value, 'Float32Array')
}

function isTypedArray(value: unknown, type: 'Uint8Array' | 'Float32Array'): boolean {
  return ArrayBuffer.isView(value) && Object.prototype.toString.call(value) === `[object ${type}]`
}
