import type {
  CountTokensOptions,
  MessageData,
  ModelStreamEvent,
  StreamOptions,
} from '@strands-agents/sdk'
import type { WebLLMModelConfig } from 'strands-webllm'

export interface HarnessRequest {
  messages: MessageData[]
  config: WebLLMModelConfig
  options: Pick<StreamOptions, 'toolSpecs' | 'toolChoice' | 'dynamicTrailingBlocks'> & {
    systemPrompt?: string
  }
}

export interface HarnessCountRequest extends Omit<HarnessRequest, 'options'> {
  options: CountTokensOptions
}

export type HarnessResult =
  IteratorResult<ModelStreamEvent, void> | { error: { name: string; message: string } }

declare global {
  interface Window {
    harness: {
      open(modelId: string | string[], contextWindow?: number): Promise<void>
      count(request: HarnessCountRequest): Promise<number>
      start(request: HarnessRequest): Promise<string>
      next(id: string): Promise<HarnessResult>
      stop(id: string): Promise<void>
      close(): Promise<void>
    }
  }
}
