import { trimThoughts, qwenHistory, nativeMessages } from './history.js'
import type {
  ChatCompletionRequestStreaming,
  ChatCompletionRequestBase,
  ChatCompletionChunk,
  Completion,
  CompletionCreateParamsStreaming,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'
import type { GemmaMessage } from './gemma/prompt.js'
import { gemmaRequest } from './gemma/request.js'
import { gemmaChunks } from './gemma/stream.js'
import { qwenChunks } from './qwen/stream.js'
import { qwen35Chunks } from './qwen35/stream.js'
import { isQwen } from './qwen/models.js'
import { isQwen35 } from './qwen35/models.js'
import { isTextFormat } from './response.js'
import { streamCall } from './engine/call.js'
import type { StreamChunk } from './stream/chunks.js'
import { nativeChunks, nativeHistory, nativeTools } from './native.js'
import type { ToolSpec, ToolChoice } from './types.js'

export type WebLLMRequest = Omit<
  ChatCompletionRequestBase,
  'messages' | 'model' | 'stream' | 'stream_options' | 'tools' | 'tool_choice' | 'n'
>
export type WebLLMChunk = ChatCompletionChunk | Completion

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming

export interface ProtocolInput {
  prepare?: <T extends NativeRequest>(request: T) => Promise<T>
  open?: <T>(run: () => Promise<AsyncIterable<T>>) => Promise<AsyncIterable<T>>
  engine: MLCEngineInterface
  modelId: string
  history: GemmaMessage[]
  tools: ToolSpec[]
  choice: ToolChoice | 'none' | undefined
  request: WebLLMRequest
  onChunk?: ((chunk: WebLLMChunk) => void) | undefined
  signal?: AbortSignal | undefined
}

export function protocolStream({
  engine,
  open = (run) => run(),
  prepare = async (request) => request,
  modelId,
  history,
  tools,
  choice,
  request,
  onChunk,
  signal,
}: ProtocolInput): AsyncIterable<StreamChunk> {
  const gemma = /^gemma-4-/i.test(modelId)
  const structured = !isTextFormat(request.response_format)
  if (choice === 'none') tools = []
  const thinking = request.extra_body?.enable_thinking === true
  if (gemma) history = trimThoughts(history)
  const gemmaProtocol =
    gemma &&
    (tools.length > 0 ||
      thinking ||
      history.some(
        (message) =>
          message.role === 'tool' ||
          (message.role === 'assistant' && (message.tool_calls?.length || message.reasoning)),
      ))
  const sampling = {
    ...request,
    model: modelId,
    stream: true as const,
    stream_options: { include_usage: true },
    n: 1,
  }
  if (gemma && request.extra_body?.enable_thinking === false) {
    sampling.extra_body = { ...sampling.extra_body }
    delete sampling.extra_body.enable_thinking
  }
  if (gemmaProtocol) {
    const { createRequest, prefix } = gemmaRequest(history, tools, choice, sampling)
    return stream(
      async () => engine.completions.create(await prepare(createRequest())),
      (chunks) =>
        gemmaChunks(chunks, tools, {
          onChunk,
          prefix,
          stopRequested: Boolean(request.stop?.length),
          signal,
        }),
    )
  }
  if (isQwen(modelId) || isQwen35(modelId)) {
    if (tools.length && structured)
      throw new Error('Qwen tool calls cannot use a structured response format.')
    if (
      modelId === 'Qwen3-1.7B-q4f16_1-MLC' &&
      (tools.length ||
        history.some(
          (message) =>
            message.role === 'tool' || (message.role === 'assistant' && message.tool_calls?.length),
        ) ||
        (choice && choice !== 'none' && !('auto' in choice)))
    )
      throw new Error('Tool calling is not supported for Qwen3-1.7B-q4f16_1-MLC.')
    const xml = isQwen35(modelId)
    const messages = qwenHistory(
      modelId,
      history,
      tools,
      choice === 'none' ? undefined : choice,
      xml,
    )
    return stream(
      async () => engine.chat.completions.create(await prepare({ ...sampling, messages })),
      (chunks) =>
        xml
          ? qwen35Chunks(chunks, tools, { onChunk, parseProtocol: !structured, signal })
          : qwenChunks(chunks, tools, {
              onChunk,
              parseProtocol: !structured,
              parseThinking: modelId.startsWith('Qwen3-'),
              signal,
            }),
    )
  }
  if (
    choice !== 'none' &&
    choice &&
    'tool' in choice &&
    !tools.some((tool) => tool.name === choice.tool.name)
  )
    throw new Error('The chosen WebLLM tool is not registered.')
  const native: ChatCompletionRequestStreaming = {
    ...sampling,
    messages: nativeHistory(nativeMessages(history), modelId, Boolean(tools.length)),
  }
  if (tools.length) native.tools = nativeTools(tools)
  if (choice === 'none') native.tool_choice = 'none'
  else if (choice && 'any' in choice)
    throw new Error('WebLLM does not support required tool choice for this model.')
  else if (choice && 'tool' in choice)
    native.tool_choice = { type: 'function', function: { name: choice.tool.name } }
  return stream(
    // Hermes adds its system prompt and response format while opening a request.
    async () =>
      engine.chat.completions.create(await prepare({ ...native, messages: [...native.messages] })),
    (chunks) => nativeChunks(chunks, Boolean(native.tools), { onChunk, signal }),
  )

  function stream<T>(
    create: () => Promise<AsyncIterable<T>>,
    read: (chunks: AsyncIterable<T>) => AsyncIterable<StreamChunk>,
  ) {
    return streamCall(engine, () => open(create), read, signal)
  }
}
