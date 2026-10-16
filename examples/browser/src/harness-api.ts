import type { MessageData, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk'
import type { WebLLMModelConfig } from 'strands-webllm'

export interface HarnessRequest {
  messages: MessageData[]
  config: WebLLMModelConfig
  options: Pick<StreamOptions, 'toolSpecs' | 'toolChoice'> & { systemPrompt?: string }
}

declare global {
  interface Window {
    harness: {
      open(modelId: string): Promise<void>
      start(request: HarnessRequest): Promise<void>
      next(): Promise<IteratorResult<ModelStreamEvent, void>>
      stop(): Promise<void>
      close(): Promise<void>
    }
  }
}
