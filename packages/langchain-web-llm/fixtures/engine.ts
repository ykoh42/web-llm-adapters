import { MLCEngine, type ChatCompletionChunk } from '@mlc-ai/web-llm'
import { vi } from 'vitest'
import { ChatWebLLM } from '../src/index.js'

export function chunk(
  delta: ChatCompletionChunk.Choice.Delta = {},
  finish: ChatCompletionChunk.Choice['finish_reason'] = null,
): ChatCompletionChunk {
  return {
    id: 'test',
    model: 'test-model',
    object: 'chat.completion.chunk',
    created: 0,
    choices: [{ index: 0, delta, finish_reason: finish }],
  }
}

export async function* chunks(...items: ChatCompletionChunk[]) {
  yield* items
}

export function setup(...items: ChatCompletionChunk[]) {
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  const create = vi
    .spyOn(engine.chat.completions, 'create')
    .mockImplementation(async () => chunks(...items))
  const stop = vi.spyOn(engine, 'interruptGenerate').mockResolvedValue(undefined)
  const model = new ChatWebLLM({ engine, modelId: 'test-model' })
  return { engine, model, create, stop }
}

export async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = []
  for await (const item of stream) items.push(item)
  return items
}

export function pause() {
  let resume!: () => void
  const wait = new Promise<void>((resolve) => {
    resume = resolve
  })
  return { wait, resume }
}
