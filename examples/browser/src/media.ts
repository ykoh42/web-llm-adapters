import { Agent, AudioBlock, ImageBlock, TextBlock } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-webllm'
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
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    contextWindowLimit: 8192,
    maxTokens: 256,
    temperature: 0,
  })
  await runAgent(new Agent({ model, printer: false }), [new TextBlock(input.prompt), block], input)
}
