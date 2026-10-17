import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type {
  ChatCompletionChunk,
  ChatCompletionRequestStreaming,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from './types.js'
import { generationOptions } from './request.js'
import { formatMessages, formatTools, formatToolChoice } from './messages.js'
import { streamCall } from './call.js'
import { readStream } from './stream.js'
import { stopReason, type StreamChunk } from './chunks.js'

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
    (chunks) =>
      readStream(
        chunks,
        (chunk) => chatChunk(chunk, Boolean(request.tools)),
        options.cancelSignal,
        onChunk,
      ),
    options.cancelSignal,
  )
}

function chatChunk(chunk: ChatCompletionChunk, hasTools: boolean): StreamChunk {
  const choice = chunk.choices[0]
  if (choice && (chunk.choices.length > 1 || choice.index !== 0))
    throw new ModelError('Strands needs one response per request.')
  const text = choice?.delta.content ?? ''
  return {
    text: hasTools ? '' : text,
    reasoning: '',
    // Hermes streams its JSON text before its parsed calls arrive.
    toolText: hasTools ? text : '',
    calls: (choice?.delta.tool_calls ?? []).map((call) => ({
      index: call.index,
      id: call.id,
      name: call.function?.name,
      input: call.function?.arguments,
    })),
    stop: stopReason(choice?.finish_reason),
    usage: chunk.usage,
  }
}
