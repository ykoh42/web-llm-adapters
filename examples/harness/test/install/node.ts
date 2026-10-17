import { createHarness } from '@strands-agents/harness'
import { BrowserModel, type BrowserPage, type WebLLMModelConfig } from 'strands-webllm'

// Check the documented Node.js API without importing browser-native protocol types.
export async function connect(page: BrowserPage) {
  const model = new BrowserModel(page, {
    modelId: 'loaded-model',
    maxTokens: 256,
    request: { seed: 7, logprobs: true },
    onChunk: (chunk) => console.log(chunk.usage),
  })
  const config: WebLLMModelConfig = model.getConfig()
  const agent = await createHarness({ model, printer: false })
  return { agent, config }
}
