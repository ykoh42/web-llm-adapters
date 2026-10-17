import {
  Model,
  type CountTokensOptions,
  type Message,
  type StreamOptions,
} from '@strands-agents/sdk'
import type { ModelStreamEvent } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { chatStream } from './chat.js'
import { isGemma, usesGemmaProtocol } from './gemma/request.js'
import { gemmaStream } from './gemma/model.js'
import { promptTokens } from './gemma/prompt.js'
import { trimThoughts } from './gemma/history.js'
import type { WebLLMModelConfig, WebLLMModelOptions } from './types.js'

/** Connects a loaded WebLLM engine to Strands. The caller owns the engine. */
export class WebLLMModel extends Model<WebLLMModelConfig> {
  readonly engine: MLCEngineInterface
  private config: WebLLMModelConfig
  private readonly onChunk: WebLLMModelOptions['onChunk']

  constructor({ engine, onChunk, ...config }: WebLLMModelOptions) {
    super()
    this.engine = engine
    this.onChunk = onChunk
    this.config = structuredClone(config)
  }

  override getConfig(): WebLLMModelConfig {
    return structuredClone(this.config)
  }

  /** Selects a loaded model. Load new models with engine.reload() first. */
  override updateConfig(config: Partial<WebLLMModelConfig>): void {
    this.config = structuredClone({ ...this.config, ...config })
  }

  override async countTokens(
    messages: Message[],
    options: CountTokensOptions = {},
  ): Promise<number> {
    if (!isGemma(this.config.modelId)) return super.countTokens(messages, options)
    const thinking = this.config.request?.extra_body?.enable_thinking === true
    return (await super.countTokens(trimThoughts(messages))) + promptTokens(options, thinking)
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const signal = options.cancelSignal
    signal?.throwIfAborted()
    const config = this.getConfig()
    const stream = usesGemmaProtocol(config, messages, options) ? gemmaStream : chatStream
    yield* stream(this.engine, config, messages, options, this.onChunk)
  }
}
