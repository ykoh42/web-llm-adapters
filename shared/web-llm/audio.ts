const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1
// Match WebLLM's WAV and PCM input limit.
const maxSampleRate = 384_000

export function checkSampleRate(sampleRate: number) {
  if (!Number.isSafeInteger(sampleRate) || sampleRate <= 0 || sampleRate > maxSampleRate)
    throw new RangeError(`Sample rate must be a positive integer no higher than ${maxSampleRate}.`)
}

/** Encode mono Float32 PCM as WAV without changing samples or sample rate. */
export function pcmWav(samples: Float32Array, sampleRate: number): Uint8Array {
  checkSampleRate(sampleRate)
  const count = samples.length
  if (!count) throw new RangeError('Audio must contain finite samples.')
  for (let index = 0; index < count; index++) {
    if (!Number.isFinite(samples[index])) throw new RangeError('Audio must contain finite samples.')
  }

  const bytes = new Uint8Array(56 + samples.byteLength)
  const view = new DataView(bytes.buffer)
  const label = (offset: number, text: string) => {
    bytes.set(new TextEncoder().encode(text), offset)
  }
  label(0, 'RIFF')
  view.setUint32(4, bytes.length - 8, true)
  label(8, 'WAVE')
  label(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 3, true) // IEEE float
  view.setUint16(22, 1, true) // Mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 4, true)
  view.setUint16(32, 4, true)
  view.setUint16(34, 32, true)
  label(36, 'fact')
  view.setUint32(40, 4, true)
  view.setUint32(44, samples.length, true)
  label(48, 'data')
  view.setUint32(52, samples.byteLength, true)
  if (littleEndian)
    bytes.set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength), 56)
  else
    for (let index = 0; index < count; index++)
      view.setFloat32(56 + index * 4, samples[index]!, true)
  return bytes
}
