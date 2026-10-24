import { WebLLMModel } from '../src/index.js'
import { setupEngine } from '../../../fixtures/web-llm.js'

export { chunk, chunks, collect, pause } from '../../../fixtures/web-llm.js'

export function setup(...items: Parameters<typeof setupEngine>) {
  const { engine, create, stop } = setupEngine(...items)
  const model = new WebLLMModel({ engine, modelId: 'test-model' })
  return { engine, model, create, stop }
}
