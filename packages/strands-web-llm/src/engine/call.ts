import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { ModelStreamEvent } from '@strands-agents/sdk'
import { streamCall as readCall } from '../../../../shared/web-llm/engine/call.js'
import { modelError } from './error.js'

export async function* streamCall<T>(
  engine: MLCEngineInterface,
  open: () => Promise<AsyncIterable<T>>,
  read: (chunks: AsyncIterable<T>) => AsyncIterable<ModelStreamEvent>,
  signal?: AbortSignal,
): AsyncGenerator<ModelStreamEvent> {
  try {
    yield* readCall(engine, open, read, signal)
  } catch (error) {
    throw modelError(error)
  }
}
