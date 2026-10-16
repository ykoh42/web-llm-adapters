import type { BaseModelConfig } from '@strands-agents/sdk'
import type {
  ChatCompletionChunk,
  Completion,
  ChatCompletionRequestBase,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'

/** Actual native chunks: chat requests or Gemma raw completions. */
export type WebLLMChunk = ChatCompletionChunk | Completion

/** Native options that do not change the Strands message or tool contract. */
export type WebLLMRequest = Omit<
  ChatCompletionRequestBase,
  'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'tool_choice' | 'n'
>

export interface WebLLMModelConfig extends BaseModelConfig {
  modelId: string
  request?: WebLLMRequest
}

export interface WebLLMModelOptions extends WebLLMModelConfig {
  engine: MLCEngineInterface
  /** Receives native chunks, including log probabilities and WebLLM metadata. */
  onChunk?: (chunk: WebLLMChunk) => void
}
