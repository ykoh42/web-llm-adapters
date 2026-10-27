import type { SimpleStreamOptions } from '@earendil-works/pi-ai'
import type { ChatCompletionToolChoiceOption, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { ChunkCallback } from '../../../shared/web-llm/callback.js'
import type { WebLLMChunk, WebLLMRequest } from '../../../shared/web-llm/protocol.js'

export type { WebLLMChunk, WebLLMRequest }
export type WebLLMApi = 'webllm'

export interface WebLLMProviderOptions {
  /** A caller-owned MLCEngine or WebWorkerMLCEngine with modelId already loaded. */
  engine: MLCEngineInterface
  modelId: string
  /** The loaded Qwen template already supplies the opening thinking tag. */
  thinkingPrefilled?: boolean
  /** Match context_window_size used when loading the engine. */
  contextWindow: number
  maxTokens?: number
  providerId?: string
  name?: string
  /** Defaults to true for Qwen3, Qwen3.5, and Gemma 4. */
  reasoning?: boolean
  /** Defaults on for built-in Phi-3.5 Vision models; set for custom artifacts or false to disable. */
  imageInputs?: boolean
  request?: WebLLMRequest
  /** Receives native chunks during inference; don't await work on the same engine. */
  onChunk?: ChunkCallback<WebLLMChunk>
}

export interface WebLLMStreamOptions extends Omit<SimpleStreamOptions, 'toolChoice'> {
  /** Select a registered function using WebLLM's native tool-choice format. */
  toolChoice?: ChatCompletionToolChoiceOption
  request?: WebLLMRequest
}

declare module '@earendil-works/pi-ai' {
  interface ApiOptionsMap {
    webllm: WebLLMStreamOptions
  }
}
