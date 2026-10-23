import {
  type JSONValue,
  type Message,
  type ModelStreamEvent,
  type StreamOptions,
} from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { chatStream } from './chat/model.js'
import { modelError } from './engine/error.js'
import { isGemma, usesGemmaProtocol } from './gemma/request.js'
import { gemmaStream } from './gemma/model.js'
import { trimThoughts } from './gemma/history.js'
import { usesQwenProtocol } from './qwen/request.js'
import { qwenStream } from './qwen/model.js'
import { usesQwen35Protocol } from './qwen35/request.js'
import type { WebLLMChunk, WebLLMModelOptions } from './types.js'
import { WebLLMModelBase } from './model-base.js'
import { preserveThinking } from '../../../shared/web-llm/request.js'
import { appendLogprobs, type WebLLMLogprobs } from '../../../shared/web-llm/logprobs.js'

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
    const preserve = preserveThinking(config.modelId, config.request ?? {})
    options = {
      ...options,
      ...(options.toolSpecs && { toolSpecs: structuredClone(options.toolSpecs) }),
      ...(options.toolChoice && { toolChoice: structuredClone(options.toolChoice) }),
    }
    // Past-turn thoughts must not force later JSON or audio requests onto raw completions.
    if (isGemma(config.modelId))
      messages = trimThoughts(messages, options.dynamicTrailingBlocks, preserve)
    if (config.request?.tool_choice === 'none' && !options.toolChoice)
      options = { ...options, toolSpecs: [] }
    let stream = chatStream
    if (usesGemmaProtocol(config, messages, options)) stream = gemmaStream
    else if (usesQwen35Protocol(config) || usesQwenProtocol(config)) stream = qwenStream
    let logprobs: WebLLMLogprobs | undefined
    try {
      const onChunk =
        config.request?.logprobs === true
          ? (chunk: WebLLMChunk) => {
              this.onChunk?.(chunk)
              logprobs = appendLogprobs(logprobs, chunk.choices[0]?.logprobs)
            }
          : this.onChunk
      const chunks = stream(this.engine, config, messages, options, onChunk)
      for await (const event of chunks) {
        if (event.type === 'modelMessageStopEvent' && logprobs) {
          yield {
            ...event,
            additionalModelResponseFields: {
              // Preserve the native fields, including non-finite log probabilities.
              logprobs: jsonLogprobs(logprobs),
            },
          }
        } else yield event
      }
    } catch (error) {
      if (signal?.aborted) throw error
      throw modelError(error)
    }
  }
}

function jsonLogprobs(logprobs: WebLLMLogprobs): JSONValue {
  return {
    content:
      logprobs.content?.map(({ token, bytes, logprob, top_logprobs }) => ({
        token,
        bytes,
        logprob,
        top_logprobs: top_logprobs.map(({ token, bytes, logprob }) => ({
          token,
          bytes,
          logprob,
        })),
      })) ?? null,
  }
}
