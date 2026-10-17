import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { streamCall } from '../call.js'
import { readStream } from '../stream.js'
import { gemmaRequest } from './request.js'
import { gemmaChunks } from './stream.js'

/** Gemma's native thinking and tool protocol through raw completions. */
export function gemmaStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
) {
  const { request, prefix } = gemmaRequest(config, messages, options)
  return streamCall(
    engine,
    () => engine.completions.create(request),
    (chunks) =>
      readStream(
        gemmaChunks(chunks, options.toolSpecs ?? [], onChunk, prefix),
        false,
        options.cancelSignal,
      ),
    options.cancelSignal,
  )
}
