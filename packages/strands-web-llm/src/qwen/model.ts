import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import { readStream } from '../stream/read.js'
import { hasStructuredResponse } from '../request.js'
import { qwenRequest } from './request.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { usesQwen35Protocol, qwen35Request } from '../qwen35/request.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'

/** Qwen uses its documented tool tags because WebLLM exposes no tool API for these IDs. */
export function qwenStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
) {
  const xml = usesQwen35Protocol(config)
  const createRequest = xml ? qwen35Request : qwenRequest
  const parseChunks = xml ? qwen35Chunks : qwenChunks
  const request = createRequest(config, messages, options)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) =>
      readStream(
        parseChunks(chunks, options.toolSpecs ?? [], {
          onChunk,
          parseProtocol: !hasStructuredResponse(config),
          thinkingDisabled: request.extra_body?.enable_thinking === false,
          thinkingPrefilled:
            config.thinkingPrefilled === true && request.extra_body?.enable_thinking !== false,
          ...(!xml && { parseThinking: config.modelId.startsWith('Qwen3-') }),
          stopRequested: Boolean(request.stop?.length),
          signal: options.cancelSignal,
        }),
        options.cancelSignal,
        options.toolChoice ?? config.request?.tool_choice,
      ),
    options.cancelSignal,
  )
}
