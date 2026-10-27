import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { generationOptions } from '../request.js'
import { formatMessages, formatToolChoice } from './messages.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import { readStream } from '../stream/read.js'
import { nativeChunks, nativeHistory, nativeTools } from '../../../../shared/web-llm/native.js'
import { singleToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { qwenToolResponseFormat } from '../../../../shared/web-llm/qwen/constraints.js'

/** Native WebLLM chat, including its Hermes tool format. */
export function chatStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
  onChunkMayMutate = true,
) {
  const choice = options.toolChoice && singleToolChoice(options.toolChoice, options.toolSpecs ?? [])
  if (qwenToolResponseFormat(config.modelId, config.request ?? {}, options.toolSpecs ?? [], choice))
    throw new ModelError('Constrained tool calls require the Qwen protocol.')
  if (choice && 'tool' in choice) {
    const name = choice.tool.name
    if (!options.toolSpecs?.some((tool) => tool.name === name))
      throw new ModelError('The chosen WebLLM tool is not registered.')
  }

  const request: ChatCompletionRequestStreaming = {
    ...generationOptions(config),
    messages: nativeHistory(
      formatMessages(messages, options.systemPrompt),
      config.modelId,
      Boolean(options.toolSpecs?.length),
    ),
  }
  if (options.toolSpecs?.length) request.tools = nativeTools(options.toolSpecs)
  if (choice) request.tool_choice = formatToolChoice(choice)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) =>
      readStream(
        nativeChunks(chunks, Boolean(request.tools), {
          onChunk,
          onChunkMayMutate,
          signal: options.cancelSignal,
        }),
        options.cancelSignal,
        choice ?? config.request?.tool_choice,
        options.toolSpecs?.map((tool) => tool.name) ?? [],
        true,
      ),
    options.cancelSignal,
  )
}
