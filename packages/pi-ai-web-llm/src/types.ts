import type { SimpleStreamOptions } from '@earendil-works/pi-ai'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMChunk, WebLLMRequest } from '../../../shared/web-llm/protocol.js'

export type { WebLLMChunk, WebLLMRequest }
export type WebLLMApi = 'webllm'

export interface WebLLMProviderOptions {
  /** A caller-owned MLCEngine or WebWorkerMLCEngine with modelId already loaded. */
  engine: MLCEngineInterface
  modelId: string
  /** Match context_window_size used when loading the engine. */
  contextWindow: number
  maxTokens?: number
  providerId?: string
  name?: string
  /** Defaults to true for Qwen3, Qwen3.5, and Gemma 4. */
  reasoning?: boolean
  /** Enable only when the loaded artifact supports images. */
  imageInputs?: boolean
  request?: WebLLMRequest
  onChunk?: (chunk: WebLLMChunk) => void
}

export interface WebLLMStreamOptions extends SimpleStreamOptions {
  request?: WebLLMRequest
}

declare module '@earendil-works/pi-ai' {
  interface ApiOptionsMap {
    webllm: WebLLMStreamOptions
  }
}
