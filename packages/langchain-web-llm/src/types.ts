import type { ModelProfile } from '@langchain/core/language_models/profile'
import type {
  BaseChatModelCallOptions,
  BaseChatModelParams,
  BindToolsInput,
} from '@langchain/core/language_models/chat_models'
import type { MLCEngineInterface, ChatCompletionChunk, Completion } from '@mlc-ai/web-llm'
import type { WebLLMRequest } from '../../../shared/web-llm/request.js'

export type WebLLMChunk = ChatCompletionChunk | Completion
export type { WebLLMRequest }

export interface WebLLMModelOptions extends BaseChatModelParams {
  modelId: string
  /** The loaded Qwen template already supplies the opening thinking tag. */
  thinkingPrefilled?: boolean
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
  /** LangChain agents pass provider schemas through this standard option. */
  response_format?:
    | NonNullable<WebLLMRequest['response_format']>
    | {
        type: 'json_schema'
        json_schema: {
          name?: string
          description?: string
          schema: Record<string, unknown>
          strict?: boolean
        }
      }
}
