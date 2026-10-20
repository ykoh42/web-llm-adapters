import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { streamCall } from '../engine/call.js'
import { readStream } from '../stream/read.js'
import { hasStructuredResponse } from '../request.js'
import { qwen35Request } from './request.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'

/** Qwen3.5 uses its native XML function format through WebLLM chat completions. */
export function qwen35Stream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
) {
  const request = qwen35Request(config, messages, options)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) =>
      readStream(
        qwen35Chunks(chunks, options.toolSpecs ?? [], {
          onChunk,
          parseProtocol: !hasStructuredResponse(config),
          signal: options.cancelSignal,
        }),
        options.cancelSignal,
        options.toolChoice ?? config.request?.tool_choice,
      ),
    options.cancelSignal,
  )
}
