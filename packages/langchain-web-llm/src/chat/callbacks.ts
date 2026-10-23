import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'

export function hasTokenCallbacks(runManager: CallbackManagerForLLMRun | undefined): boolean {
  return Boolean(
    runManager?.handlers.some((handler) => !handler.ignoreLLM && handler.handleLLMNewToken),
  )
}
