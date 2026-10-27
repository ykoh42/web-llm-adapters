import { Model, type CountTokensOptions, type Message } from '@strands-agents/sdk'
import { modelError } from './engine/error.js'
import { promptTokens } from './gemma/prompt.js'
import { trimThoughts } from './gemma/history.js'
import { trimThoughts as trimQwenThoughts } from './qwen/history.js'
import { isGemma, isQwen, isQwen35, protocolModelId } from '../../../shared/web-llm/models.js'
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
    checkThinkingPrefill(
      protocolModelId(this.config.modelId, this.config.sourceModelId),
      this.config.thinkingPrefilled,
    )
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
    if (
      config.modelId !== undefined &&
      config.modelId !== this.config.modelId &&
      this.config.sourceModelId !== undefined &&
      !Object.hasOwn(config, 'sourceModelId')
    )
      throw new Error('Set or clear sourceModelId when changing a custom model ID.')
    const next = structuredClone({ ...this.config, ...config })
    if (!next.modelId) throw new Error('modelId must name a loaded WebLLM model.')
    checkRequest(next.request ?? {}, 'none')
    checkThinkingPrefill(protocolModelId(next.modelId, next.sourceModelId), next.thinkingPrefilled)
    this.config = next
  }

  override async countTokens(
    messages: Message[],
    options: CountTokensOptions = {},
  ): Promise<number> {
    const read = this.getConfig
    const config = read === defaultGetConfig ? this.config : read.call(this)
    const source = protocolModelId(config.modelId, config.sourceModelId)
    const preserve = preserveThinking(source, config.request ?? {})
    if (isQwen(source) || isQwen35(source))
      return super.countTokens(trimQwenThoughts(messages, !source.startsWith('Qwen2.5-')), options)
    if (!isGemma(source)) return super.countTokens(messages, options)
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
