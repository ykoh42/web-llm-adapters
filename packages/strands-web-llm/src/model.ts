import { type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ModelStreamEvent } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { chatStream } from './chat/model.js'
import { modelError } from './engine/error.js'
import { isGemma, usesGemmaProtocol } from './gemma/request.js'
import { gemmaStream } from './gemma/model.js'
import { trimThoughts } from './gemma/history.js'
import { usesQwenProtocol } from './qwen/request.js'
import { qwenStream } from './qwen/model.js'
import { usesQwen35Protocol } from './qwen35/request.js'
import { qwen35Stream } from './qwen35/model.js'
import type { WebLLMModelOptions } from './types.js'
import { WebLLMModelBase } from './model-base.js'

/** Connects a loaded WebLLM engine to Strands. The caller owns the engine. */
export class WebLLMModel extends WebLLMModelBase {
  readonly engine: MLCEngineInterface
  private readonly onChunk: WebLLMModelOptions['onChunk']

  constructor({ engine, onChunk, ...config }: WebLLMModelOptions) {
    super(config)
    this.engine = engine
    this.onChunk = onChunk
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const signal = options.cancelSignal
    signal?.throwIfAborted()
    const config = this.getConfig()
    options = {
      ...options,
      ...(options.toolSpecs && { toolSpecs: structuredClone(options.toolSpecs) }),
      ...(options.toolChoice && { toolChoice: structuredClone(options.toolChoice) }),
    }
    // Past-turn thoughts must not force later JSON or audio requests onto raw completions.
    if (isGemma(config.modelId)) {
      messages = trimThoughts(messages, options.dynamicTrailingBlocks)
      // WebLLM's native false prefix is Qwen-specific; Gemma disables thinking without it.
      if (config.request?.extra_body?.enable_thinking === false)
        delete config.request.extra_body.enable_thinking
    }
    if (config.request?.tool_choice === 'none' && !options.toolChoice)
      options = { ...options, toolSpecs: [] }
    let stream = chatStream
    if (usesGemmaProtocol(config, messages, options)) stream = gemmaStream
    else if (usesQwen35Protocol(config)) stream = qwen35Stream
    else if (usesQwenProtocol(config)) stream = qwenStream
    try {
      yield* stream(this.engine, config, messages, options, this.onChunk)
    } catch (error) {
      throw modelError(error)
    }
  }
}
