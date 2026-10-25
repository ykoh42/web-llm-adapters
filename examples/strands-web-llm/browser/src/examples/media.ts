import { sampling } from '../../../sampling.js'
import { Agent, AudioBlock, ImageBlock, TextBlock } from '@strands-agents/sdk'
import { WebLLMModel, pcmAudio } from 'strands-web-llm'
import { runAgent, type ExampleInput } from './output.js'

export async function media(input: ExampleInput) {
  if (!input.file) throw new Error('Choose an image or WAV file first.')
  const bytes = new Uint8Array(await input.file.arrayBuffer())
  input.signal.throwIfAborted()
  let block: ImageBlock | AudioBlock
  switch (input.file.type) {
    case 'image/png':
      block = new ImageBlock({ format: 'png', source: { bytes } })
      break
    case 'image/jpeg':
      block = new ImageBlock({ format: 'jpeg', source: { bytes } })
      break
    case 'audio/wav':
    case 'audio/x-wav':
    case 'audio/wave':
      block = new AudioBlock({ format: 'wav', source: { bytes } })
      break
    default:
      throw new Error('Use a PNG, JPEG, or WAV file.')
  }
  await sendMedia(block, input)
}

function sendMedia(block: ImageBlock | AudioBlock, input: ExampleInput) {
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    thinkingPrefilled: input.modelId.startsWith('Qwen3.5-'),
    contextWindowLimit: 8192,
    maxTokens: 256,
    ...sampling(input.modelId, false),
  })
  return runAgent(new Agent({ model, printer: false }), [new TextBlock(input.prompt), block], input)
}

export async function rawAudio(input: ExampleInput) {
  if (!input.file) throw new Error('Choose an audio file first.')
  const audio = new AudioContext({ sampleRate: 16000 })
  try {
    const decoded = await audio.decodeAudioData(await input.file.arrayBuffer())
    input.signal.throwIfAborted()
    const samples = new Float32Array(decoded.length)
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const values = decoded.getChannelData(channel)
      for (let i = 0; i < samples.length; i++) samples[i]! += values[i]! / decoded.numberOfChannels
    }
    await sendMedia(pcmAudio(samples, decoded.sampleRate), input)
  } finally {
    await audio.close()
  }
}
