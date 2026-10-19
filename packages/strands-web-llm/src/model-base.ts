import { Model, type CountTokensOptions, type Message } from '@strands-agents/sdk'
import { modelError } from './engine/error.js'
import { isGemma } from './gemma/request.js'
import { promptTokens } from './gemma/prompt.js'
import { trimThoughts } from './gemma/history.js'
import type { WebLLMModelConfig } from './types.js'

/** Shared Strands configuration and token counting for browser and local models. */
export abstract class WebLLMModelBase extends Model<WebLLMModelConfig> {
  private config: WebLLMModelConfig

  constructor(config: WebLLMModelConfig) {
    super()
    this.config = structuredClone(config)
  }

  override getConfig(): WebLLMModelConfig {
    return structuredClone(this.config)
  }

  override updateConfig(config: Partial<WebLLMModelConfig>): void {
    this.config = structuredClone({ ...this.config, ...config })
  }

  override async countTokens(
    messages: Message[],
    options: CountTokensOptions = {},
  ): Promise<number> {
    if (!isGemma(this.config.modelId)) return super.countTokens(messages, options)
    const thinking = this.config.request?.extra_body?.enable_thinking === true
    try {
      return (await super.countTokens(trimThoughts(messages))) + promptTokens(options, thinking)
    } catch (error) {
      throw modelError(error)
    }
  }
}
