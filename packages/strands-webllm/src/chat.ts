import type { Message, StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming, MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from './types.js'
import { generationOptions } from './request.js'
import { formatMessages, formatTools, formatToolChoice } from './messages.js'
import { streamCall } from './call.js'
import { readStream } from './stream.js'

/** Native WebLLM chat, including its Hermes tool format. */
export function chatStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: WebLLMModelOptions['onChunk'],
) {
  const request: ChatCompletionRequestStreaming = {
    ...generationOptions(config),
    messages: formatMessages(messages, options.systemPrompt),
  }
  if (options.toolSpecs?.length) request.tools = formatTools(options.toolSpecs)
  if (options.toolChoice) request.tool_choice = formatToolChoice(options.toolChoice)
  return streamCall(
    engine,
    () => engine.chat.completions.create(request),
    (chunks) => readStream(chunks, Boolean(request.tools), options.cancelSignal, onChunk),
    options.cancelSignal,
  )
}
