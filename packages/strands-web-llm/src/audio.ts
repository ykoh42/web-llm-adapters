import { AudioBlock } from '@strands-agents/sdk'
import { pcmWav } from '../../../shared/web-llm/audio.js'

/** Put mono float32 PCM in a Strands audio block, preserving samples and sample rate. */
export function pcmAudio(samples: Float32Array, sampleRate: number): AudioBlock {
  return new AudioBlock({ format: 'wav', source: { bytes: pcmWav(samples, sampleRate) } })
}
