import type { BaseModelConfig } from '@strands-agents/sdk'
import type { ChatCompletionChunk, Completion, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMRequest as NativeRequest } from '../../../shared/web-llm/request.js'

/** Actual native chunks: chat requests or Gemma raw completions. */
export type WebLLMChunk = ChatCompletionChunk | Completion

/** Native options that do not change the Strands message or tool contract. */
export type WebLLMRequest = NativeRequest & {
  /** Disable tools unless Strands supplies an explicit tool choice for this call. */
  tool_choice?: 'none'
}

export interface WebLLMModelConfig extends BaseModelConfig {
  modelId: string
  /** The loaded Qwen template already supplies the opening thinking tag. */
  thinkingPrefilled?: boolean
  request?: WebLLMRequest
}

export interface WebLLMModelOptions extends WebLLMModelConfig {
  engine: MLCEngineInterface
  /** Receives native chunks, including log probabilities and WebLLM metadata. */
  onChunk?: (chunk: WebLLMChunk) => void
}
