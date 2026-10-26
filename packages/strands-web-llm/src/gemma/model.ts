import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import { readStream } from '../stream/read.js'
import { gemmaRequest } from './request.js'
import { gemmaChunks } from '../../../../shared/web-llm/gemma/stream.js'

/** Gemma's native thinking and tool protocol through raw completions. */
export function gemmaStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
  onChunkMayMutate = true,
) {
  const { request, prefix } = gemmaRequest(config, messages, options)
  return streamCall(
    engine,
    () => engine.completions.create(request),
    (chunks) =>
      readStream(
        gemmaChunks(chunks, options.toolSpecs ?? [], {
          onChunk,
          onChunkMayMutate,
          prefix,
          stopRequested: Boolean(request.stop?.length),
          signal: options.cancelSignal,
        }),
        options.cancelSignal,
        options.toolChoice ?? config.request?.tool_choice,
      ),
    options.cancelSignal,
  )
}
