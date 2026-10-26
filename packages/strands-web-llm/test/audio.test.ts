import { Agent, Message, TextBlock } from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { pcmAudio } from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

it('preserves float32 PCM and sample rate through Strands message storage and inference', async () => {
  const samples = new Float32Array([-1, -0.12345678, 0, 0.12345678, 1])
  const audio = pcmAudio(samples, 24000)
  const message = Message.fromJSON(new Message({ role: 'user', content: [audio] }).toJSON())
  samples.fill(0)
  const { model, create } = setup(chunk({ content: 'Audio received' }, 'stop'))
  const result = await new Agent({ model, printer: false }).invoke(message.content)
  expect(result.lastMessage.content).toEqual([new TextBlock('Audio received')])
  const content = create.mock.calls[0]![0].messages[0]!.content
  if (!Array.isArray(content)) throw new Error('Expected media content')
  const part = content.find((part) => part.type === 'input_audio')
  if (part?.input_audio.format !== 'wav') throw new Error('Expected WAV audio')
  const bytes = Uint8Array.from(atob(part.input_audio.data), (char) => char.charCodeAt(0))
  const view = new DataView(bytes.buffer)
  expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe('RIFF')
  expect(view.getUint32(4, true)).toBe(bytes.length - 8)
  expect(view.getUint16(20, true)).toBe(3)
  expect(view.getUint16(22, true)).toBe(1)
  expect(view.getUint32(24, true)).toBe(24000)
  expect(view.getUint32(44, true)).toBe(5)
  const decoded = Array.from({ length: 5 }, (_, i) => view.getFloat32(56 + i * 4, true))
  expect(decoded).toEqual([...new Float32Array([-1, -0.12345678, 0, 0.12345678, 1])])
})

it('preserves subarray bounds, negative zero, and subnormal samples as little-endian bytes', () => {
  const samples = new Float32Array([9, -0, 1e-45, 1 / 3, -1e-45, 8]).subarray(1, 5)
  const audio = pcmAudio(samples, 16000)
  if (audio.source.type !== 'audioSourceBytes') throw new Error('Expected audio bytes')

  expect([...audio.source.bytes.slice(56)]).toEqual([
    0, 0, 0, 128, 1, 0, 0, 0, 171, 170, 170, 62, 1, 0, 0, 128,
  ])
  expect(samples).toEqual(new Float32Array([-0, 1e-45, 1 / 3, -1e-45]))
})

it.each([0, -1, 1.5, NaN, Infinity, 0x40000000])('rejects invalid sample rate %s', (rate) => {
  expect(() => pcmAudio(new Float32Array([0]), rate)).toThrow('Sample rate')
})

it('accepts WebLLM’s maximum sample rate', () => {
  const audio = pcmAudio(new Float32Array([0]), 384_000)
  if (audio.source.type !== 'audioSourceBytes') throw new Error('Expected audio bytes')

  const bytes = audio.source.bytes
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  expect(view.getUint32(24, true)).toBe(384_000)
})

it('rejects sample rates above WebLLM’s maximum', () => {
  expect(() => pcmAudio(new Float32Array([0]), 384_001)).toThrow('384000')
})

it.each([[], [NaN], [Infinity], [-Infinity]].map((samples) => ({ samples })))(
  'rejects invalid samples $samples',
  ({ samples }) => {
    expect(() => pcmAudio(new Float32Array(samples), 16000)).toThrow('finite samples')
  },
)
