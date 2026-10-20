import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { generationOptions } from '../request.js'
import { formatMessages, formatToolChoice } from './messages.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import { readStream } from '../stream/read.js'
import { nativeChunks, nativeHistory, nativeTools } from '../../../../shared/web-llm/native.js'

/** Native WebLLM chat, including its Hermes tool format. */
export function chatStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
) {
  const choice = options.toolChoice
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
  if (options.toolChoice) request.tool_choice = formatToolChoice(options.toolChoice)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) =>
      readStream(
        nativeChunks(chunks, Boolean(request.tools), { onChunk, signal: options.cancelSignal }),
        options.cancelSignal,
        options.toolChoice ?? config.request?.tool_choice,
        options.toolSpecs?.map((tool) => tool.name),
        true,
      ),
    options.cancelSignal,
  )
}
