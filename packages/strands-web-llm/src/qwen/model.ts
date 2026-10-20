import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { streamCall } from '../engine/call.js'
import { readStream } from '../stream/read.js'
import { hasStructuredResponse } from '../request.js'
import { qwenRequest } from './request.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'

/** Qwen uses its documented tool tags because WebLLM exposes no tool API for these IDs. */
export function qwenStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
) {
  const request = qwenRequest(config, messages, options)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) =>
      readStream(
        qwenChunks(chunks, options.toolSpecs ?? [], {
          onChunk,
          parseProtocol: !hasStructuredResponse(config),
          parseThinking: config.modelId.startsWith('Qwen3-'),
          signal: options.cancelSignal,
        }),
        options.cancelSignal,
        options.toolChoice ?? config.request?.tool_choice,
      ),
    options.cancelSignal,
  )
}
