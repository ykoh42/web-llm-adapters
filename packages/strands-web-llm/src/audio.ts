import { AudioBlock } from '@strands-agents/sdk'

/** Put mono float32 PCM in a Strands audio block, preserving samples and sample rate. */
export function pcmAudio(samples: Float32Array, sampleRate: number): AudioBlock {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0 || sampleRate > 0x3fffffff)
    throw new RangeError('Sample rate must be a positive integer that fits a WAV header.')
  if (!samples.length || samples.some((sample) => !Number.isFinite(sample)))
    throw new RangeError('Audio must contain finite samples.')

  // IEEE float WAV carries the sample rate that Strands AudioBlock cannot store separately.
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
  const count = samples.length
  for (let index = 0; index < count; index++) view.setFloat32(56 + index * 4, samples[index]!, true)
  return new AudioBlock({ format: 'wav', source: { bytes } })
}
