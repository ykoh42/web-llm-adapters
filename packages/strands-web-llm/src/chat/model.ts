import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type {
  ChatCompletionChunk,
  ChatCompletionRequestStreaming,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import { generationOptions } from '../request.js'
import { formatMessages, formatTools, formatToolChoice } from './messages.js'
import { streamCall } from '../engine/call.js'
import { readStream } from '../stream/read.js'
import { stopReason, type StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'

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
        options.toolChoice ?? config.request?.tool_choice,
        options.toolSpecs?.map((tool) => tool.name),
        true,
      ),
    options.cancelSignal,
  )
}

function chatChunk(chunk: ChatCompletionChunk, hasTools: boolean): StreamChunk {
  const choice = chunk.choices[0]
  if (choice && (chunk.choices.length > 1 || choice.index !== 0))
    throw new ModelError('Strands needs one response per request.')
  const text = choice?.delta.content ?? ''
  const deltas = choice?.delta.tool_calls ?? []
  const protocolText = hasTools && (!deltas.length || isSerializedToolText(text, deltas))
  return {
    text: protocolText ? '' : text,
    reasoning: '',
    // Hermes streams its JSON text before its parsed calls arrive.
    toolText: protocolText ? text : '',
    calls: deltas.map((call) => ({
      index: call.index,
      id: call.id,
      name: call.function?.name,
      input: call.function?.arguments,
    })),
    stop: stopReason(choice?.finish_reason),
    usage: chunk.usage,
  }
}

function isSerializedToolText(
  text: string,
  calls: NonNullable<ChatCompletionChunk.Choice['delta']['tool_calls']>,
) {
  try {
    const output: unknown = JSON.parse(text)
    if (!Array.isArray(output) || output.length !== calls.length) return false

    const ordered = [...calls].sort((left, right) => left.index - right.index)
    return output.every((item, index) => {
      if (!isRecord(item)) return false
      const call = ordered[index]!
      const name = call.function?.name
      const source = call.function?.arguments
      if (typeof name !== 'string' || typeof source !== 'string' || item.name !== name) return false
      return JSON.stringify(item.arguments) === JSON.stringify(JSON.parse(source))
    })
  } catch {
    return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
