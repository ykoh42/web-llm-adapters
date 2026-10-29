import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { ChatGenerationChunk } from '@langchain/core/outputs'

export function hasTokenCallbacks(runManager: CallbackManagerForLLMRun | undefined): boolean {
  return Boolean(
    runManager?.handlers.some((handler) => !handler.ignoreLLM && handler.handleLLMNewToken),
  )
}

export function notifyToken(
  runManager: CallbackManagerForLLMRun,
  chunk: ChatGenerationChunk,
): Promise<void> {
  return runManager.handleLLMNewToken(chunk.text, undefined, undefined, undefined, undefined, {
    chunk,
  })
}
