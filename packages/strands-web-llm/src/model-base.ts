import { Model, type CountTokensOptions, type Message } from '@strands-agents/sdk'
import { modelError } from './engine/error.js'
import { isGemma } from './gemma/request.js'
import { promptTokens } from './gemma/prompt.js'
import { trimThoughts } from './gemma/history.js'
import { trimThoughts as trimQwenThoughts } from './qwen/history.js'
import { isQwen } from '../../../shared/web-llm/qwen/models.js'
import { isQwen35 } from '../../../shared/web-llm/qwen35/models.js'
import type { WebLLMModelConfig } from './types.js'
import { checkRequest, preserveThinking } from '../../../shared/web-llm/request.js'
import { checkThinkingPrefill } from '../../../shared/web-llm/qwen/prefill.js'

/** Shared Strands configuration and token counting for browser and local models. */
export abstract class WebLLMModelBase extends Model<WebLLMModelConfig> {
  private config: WebLLMModelConfig

  constructor(config: WebLLMModelConfig) {
    super()
    this.config = structuredClone(config)
    if (!this.config.modelId) throw new Error('modelId must name a loaded WebLLM model.')
    checkRequest(this.config.request ?? {}, 'none')
    checkThinkingPrefill(this.config.modelId, this.config.thinkingPrefilled)
  }

  override get modelId(): string {
    const read = this.getConfig
    // Respect custom getConfig() implementations.
    return read === defaultGetConfig ? this.config.modelId : read.call(this).modelId
  }

  override getConfig(): WebLLMModelConfig {
    return structuredClone(this.config)
  }

  override updateConfig(config: Partial<WebLLMModelConfig>): void {
    const next = structuredClone({ ...this.config, ...config })
    if (!next.modelId) throw new Error('modelId must name a loaded WebLLM model.')
    checkRequest(next.request ?? {}, 'none')
    checkThinkingPrefill(next.modelId, next.thinkingPrefilled)
    this.config = next
  }

  override async countTokens(
    messages: Message[],
    options: CountTokensOptions = {},
  ): Promise<number> {
    const read = this.getConfig
    const config = read === defaultGetConfig ? this.config : read.call(this)
    const preserve = preserveThinking(config.modelId, config.request ?? {})
    if (isQwen(config.modelId) || isQwen35(config.modelId))
      return super.countTokens(
        trimQwenThoughts(messages, !config.modelId.startsWith('Qwen2.5-')),
        options,
      )
    if (!isGemma(config.modelId)) return super.countTokens(messages, options)
    const thinking = config.request?.extra_body?.enable_thinking === true
    try {
      return (
        (await super.countTokens(trimThoughts(messages, 0, preserve))) +
        promptTokens(options, thinking)
      )
    } catch (error) {
      throw modelError(error)
    }
  }
}

const defaultGetConfig = WebLLMModelBase.prototype.getConfig
