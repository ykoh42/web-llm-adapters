import { Model, ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { BaseModelConfig, ModelStreamEvent } from '@strands-agents/sdk'
import type {
  ChatCompletionChunk,
  ChatCompletionRequestBase,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { formatMessages, formatTools, formatToolChoice } from './messages.js'
import { readStream } from './stream.js'

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
  onChunk?: (chunk: ChatCompletionChunk) => void
}

// interruptGenerate affects the engine, so two adapters must not share an active call.
const busyEngines = new WeakSet<MLCEngineInterface>()

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

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    if (busyEngines.has(this.engine)) {
      throw new ModelError('This WebLLM engine is already in use.')
    }

    const signal = options.cancelSignal
    signal?.throwIfAborted()
    busyEngines.add(this.engine)
    const stop = () => this.engine.interruptGenerate()
    let done = false

    try {
      const config = this.getConfig()
      const request: ChatCompletionRequestBase = {
        ...config.request,
        model: config.modelId,
        messages: formatMessages(messages, options.systemPrompt),
        stream: true,
        stream_options: { include_usage: true },
        n: 1,
      }
      if (config.maxTokens !== undefined) request.max_tokens = config.maxTokens
      if (config.temperature !== undefined) request.temperature = config.temperature
      if (config.topP !== undefined) request.top_p = config.topP
      if (options.toolSpecs?.length) request.tools = formatTools(options.toolSpecs)
      if (options.toolChoice) request.tool_choice = formatToolChoice(options.toolChoice)

      signal?.addEventListener('abort', stop, { once: true })
      const chunks = await this.engine.chat.completions.create({ ...request, stream: true })
      signal?.throwIfAborted()
      yield* readStream(chunks, Boolean(request.tools), signal, this.onChunk)
      done = true
    } finally {
      signal?.removeEventListener('abort', stop)
      try {
        if (!done) await stop()
      } finally {
        busyEngines.delete(this.engine)
      }
    }
  }
}
