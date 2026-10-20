import { MLCEngine, type ChatCompletionChunk } from '@mlc-ai/web-llm'
import { createModels } from '@earendil-works/pi-ai/models'
import { vi } from 'vitest'
import { createWebLLMProvider, type WebLLMProviderOptions } from '../src/index.js'

export function chunk(
  delta: ChatCompletionChunk.Choice.Delta = {},
  finish: ChatCompletionChunk.Choice['finish_reason'] = null,
): ChatCompletionChunk {
  return {
    id: 'response',
    model: 'loaded-model',
    object: 'chat.completion.chunk',
    created: 0,
    choices: [{ index: 0, delta, finish_reason: finish }],
  }
}

export async function* chunks(...items: ChatCompletionChunk[]) {
  yield* items
}

export function setup(
  items = [chunk({ content: 'Hello' }, 'stop')],
  overrides: Partial<WebLLMProviderOptions> = {},
) {
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  const create = vi
    .spyOn(engine.chat.completions, 'create')
    .mockImplementation(async () => chunks(...items))
  const interrupt = vi.spyOn(engine, 'interruptGenerate').mockResolvedValue(undefined)
  const provider = createWebLLMProvider({
    engine,
    modelId: 'loaded-model',
    contextWindow: 8192,
    ...overrides,
  })
  const model = provider.getModels()[0]!
  const models = createModels()
  models.setProvider(provider)
  return { engine, create, interrupt, provider, model, models }
}

export async function collect<T>(stream: AsyncIterable<T>) {
  const values: T[] = []
  for await (const value of stream) values.push(value)
  return values
}
