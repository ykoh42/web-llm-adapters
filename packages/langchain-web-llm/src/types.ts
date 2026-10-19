import type { ModelProfile } from '@langchain/core/language_models/profile'
import type {
  BaseChatModelCallOptions,
  BaseChatModelParams,
  BindToolsInput,
} from '@langchain/core/language_models/chat_models'
import type {
  ChatCompletionRequestBase,
  MLCEngineInterface,
  ChatCompletionChunk,
  Completion,
} from '@mlc-ai/web-llm'

export type WebLLMChunk = ChatCompletionChunk | Completion
export type WebLLMRequest = Omit<
  ChatCompletionRequestBase,
  'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'tool_choice' | 'n'
>

export interface WebLLMModelOptions extends BaseChatModelParams {
  modelId: string
  /** Set maxInputTokens to the context window used when loading the engine. */
  profile?: ModelProfile
  request?: WebLLMRequest
  temperature?: number
  maxTokens?: number
  topP?: number
  stop?: string[]
}

export interface ChatWebLLMOptions extends WebLLMModelOptions {
  engine: MLCEngineInterface
  onChunk?: (chunk: WebLLMChunk) => void
}

export interface WebLLMCallOptions extends BaseChatModelCallOptions {
  tools?: BindToolsInput[]
  request?: WebLLMRequest
}
