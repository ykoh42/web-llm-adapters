import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import { readStream } from '../stream/read.js'
import { hasStructuredResponse } from '../request.js'
import { qwenRequest } from './request.js'
import { qwenModelChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { protocolModelId } from '../../../../shared/web-llm/models.js'

/** Qwen uses its documented tool tags because WebLLM exposes no tool API for these IDs. */
export function qwenStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
  onChunkMayMutate = true,
) {
  const request = qwenRequest(config, messages, options)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) =>
      readStream(
        qwenModelChunks(
          protocolModelId(config.modelId, config.sourceModelId),
          chunks,
          options.toolSpecs ?? [],
          {
            onChunk,
            onChunkMayMutate,
            parseProtocol: !hasStructuredResponse(config),
            thinkingDisabled: request.extra_body?.enable_thinking === false,
            thinkingPrefilled:
              config.thinkingPrefilled === true && request.extra_body?.enable_thinking !== false,
            stopRequested: Boolean(request.stop?.length),
            signal: options.cancelSignal,
          },
        ),
        options.cancelSignal,
        options.toolChoice ?? config.request?.tool_choice,
      ),
    options.cancelSignal,
  )
}
