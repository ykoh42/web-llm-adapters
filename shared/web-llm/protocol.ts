import { nativeMessages } from './history.js'
import { trimThoughts } from './gemma/history.js'
import { qwenHistory } from './qwen/history.js'
import type {
  ChatCompletionRequestStreaming,
  ChatCompletionChunk,
  Completion,
  CompletionCreateParamsStreaming,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { gemmaRequest, checkGemmaRequest } from './gemma/request.js'
import { gemmaChunks } from './gemma/stream.js'
import { qwenChunks } from './qwen/stream.js'
import { qwen35Chunks } from './qwen35/stream.js'
import { isQwen } from './qwen/models.js'
import { isQwen35 } from './qwen35/models.js'
import { isTextFormat } from './response.js'
import { streamCall } from './engine/call.js'
import type { StreamChunk } from './stream/chunks.js'
import { nativeChunks, nativeHistory, nativeTools } from './native.js'
import type { ProtocolMessage, ToolSpec, ToolChoice } from './types.js'
import { nativeRequest, preserveThinking, type WebLLMRequest } from './request.js'
import { singleToolChoice } from './stream/choice.js'

export type { WebLLMRequest }
export type WebLLMChunk = ChatCompletionChunk | Completion

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming

export interface ProtocolInput {
  prepare?: <T extends NativeRequest>(request: T) => Promise<T>
  open?: <T>(run: () => Promise<AsyncIterable<T>>) => Promise<AsyncIterable<T>>
  engine: MLCEngineInterface
  modelId: string
  thinkingPrefilled?: boolean
  history: ProtocolMessage[]
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
  thinkingPrefilled = false,
  history,
  tools,
  choice,
  request,
  onChunk,
  signal,
}: ProtocolInput): AsyncIterable<StreamChunk> {
  const gemma = /^gemma-4-/i.test(modelId)
  const preserve = preserveThinking(modelId, request)
  const structured = !isTextFormat(request.response_format)
  if (choice === 'none') tools = []
  const thinking = request.extra_body?.enable_thinking === true
  if (gemma) history = trimThoughts(history, preserve)
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
    ...nativeRequest(request, modelId),
    model: modelId,
    stream: true as const,
    stream_options: { include_usage: true },
    n: 1,
  }
  if (gemmaProtocol) {
    const { createRequest, prefix } = gemmaRequest(history, tools, choice, sampling)
    let stopRequested = Boolean(request.stop?.length)
    return stream(
      async () => {
        const prepared = await prepare(createRequest())
        checkGemmaRequest(prepared)
        stopRequested = Boolean(prepared.stop?.length)
        return engine.completions.create(prepared)
      },
      (chunks) =>
        gemmaChunks(chunks, tools, {
          onChunk,
          prefix,
          stopRequested,
          signal,
        }),
    )
  }
  if (isQwen(modelId) || isQwen35(modelId)) {
    if (tools.length && structured)
      throw new Error('Qwen tool calls cannot use a structured response format.')
    const xml = isQwen35(modelId)
    const messages = qwenHistory(
      modelId,
      history,
      tools,
      choice === 'none' ? undefined : choice,
      xml,
    )
    let parseProtocol = !structured
    let prefilled = thinkingPrefilled && request.extra_body?.enable_thinking !== false
    let stopRequested = Boolean(request.stop?.length)
    return stream(
      async () => {
        const prepared = await prepare({ ...sampling, messages })
        // The payload hook may change how native output is framed.
        parseProtocol = isTextFormat(prepared.response_format)
        prefilled = thinkingPrefilled && prepared.extra_body?.enable_thinking !== false
        stopRequested = Boolean(prepared.stop?.length)
        if (tools.length && !parseProtocol)
          throw new Error('Qwen tool calls cannot use a structured response format.')
        return engine.chat.completions.create(prepared)
      },
      (chunks) =>
        xml
          ? qwen35Chunks(chunks, tools, {
              onChunk,
              parseProtocol,
              thinkingPrefilled: prefilled,
              stopRequested,
              signal,
            })
          : qwenChunks(chunks, tools, {
              onChunk,
              parseProtocol,
              parseThinking: modelId.startsWith('Qwen3-'),
              thinkingPrefilled: prefilled,
              stopRequested,
              signal,
            }),
    )
  }
  if (choice && choice !== 'none') choice = singleToolChoice(choice, tools)
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
    throw new Error('Required tool choice needs exactly one tool for this model.')
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
