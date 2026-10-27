import {
  type JSONValue,
  type Message,
  type ModelStreamEvent,
  type StreamOptions,
} from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { chatStream } from './chat/model.js'
import { modelError } from './engine/error.js'
import { usesGemmaProtocol } from './gemma/request.js'
import { gemmaStream } from './gemma/model.js'
import { trimThoughts } from './gemma/history.js'
import { isGemma, isQwen, isQwen35, protocolModelId } from '../../../shared/web-llm/models.js'
import { qwenStream } from './qwen/model.js'
import type { WebLLMChunk, WebLLMModelOptions } from './types.js'
import { WebLLMModelBase } from './model-base.js'
import { preserveThinking } from '../../../shared/web-llm/request.js'
import { appendLogprobs, type WebLLMLogprobs } from '../../../shared/web-llm/logprobs.js'
import { validateToolNames } from '../../../shared/web-llm/types.js'
import { chunkCallbackMayMutate } from '../../../shared/web-llm/callback.js'

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
    const source = protocolModelId(config.modelId, config.sourceModelId)
    const preserve = preserveThinking(source, config.request ?? {})
    options = {
      ...options,
      ...(options.toolSpecs && { toolSpecs: structuredClone(options.toolSpecs) }),
      ...(options.toolChoice && { toolChoice: structuredClone(options.toolChoice) }),
    }
    // Past-turn thoughts must not force later JSON or audio requests onto raw completions.
    if (isGemma(source)) messages = trimThoughts(messages, options.dynamicTrailingBlocks, preserve)
    if (config.request?.tool_choice === 'none' && !options.toolChoice)
      options = { ...options, toolSpecs: [] }
    let stream = chatStream
    if (usesGemmaProtocol(config, messages, options)) stream = gemmaStream
    else if (isQwen35(source) || isQwen(source)) stream = qwenStream
    let logprobs: WebLLMLogprobs | undefined
    let responseId: string | undefined
    let responseModel: string | undefined
    let responseCreated: number | undefined
    try {
      validateToolNames(options.toolSpecs ?? [])
      const onChunk = (chunk: WebLLMChunk) => {
        responseId ??= chunk.id
        responseModel ??= chunk.model
        responseCreated ??= chunk.created
        if (config.request?.logprobs === true)
          logprobs = appendLogprobs(logprobs, chunk.choices[0]?.logprobs)
        return this.onChunk?.(chunk)
      }
      const chunks = stream(
        this.engine,
        config,
        messages,
        options,
        onChunk,
        chunkCallbackMayMutate(this.onChunk),
      )
      for await (const event of chunks) {
        if (event.type !== 'modelMessageStopEvent') {
          yield event
          continue
        }
        const additionalModelResponseFields: JSONValue = {
          ...(responseId !== undefined && { id: responseId }),
          ...(responseModel !== undefined && { model: responseModel }),
          ...(responseCreated !== undefined && { created: responseCreated }),
          ...(logprobs && { logprobs: jsonLogprobs(logprobs) }),
        }
        yield Object.keys(additionalModelResponseFields).length
          ? { ...event, additionalModelResponseFields }
          : event
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
