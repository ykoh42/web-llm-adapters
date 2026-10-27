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
import { qwenModelChunks } from './qwen/stream.js'
import { isGemma, isQwen, isQwen35 } from './models.js'
import { isTextFormat } from './response.js'
import { streamCall } from './engine/call.js'
import type { StreamChunk } from './stream/chunks.js'
import type { ChunkCallback } from './callback.js'
import { nativeChunks, nativeHistory, nativeTools } from './native.js'
import { validateToolNames, type ProtocolMessage, type ToolSpec, type ToolChoice } from './types.js'
import { nativeRequest, preserveThinking, streamOptions, type WebLLMRequest } from './request.js'
import { singleToolChoice } from './stream/choice.js'
import { qwenToolResponseFormat } from './qwen/constraints.js'
import { sameJSON } from './json.js'

export type { WebLLMRequest }
export type WebLLMChunk = ChatCompletionChunk | Completion

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming

async function keepRequest<T extends NativeRequest>(request: T): Promise<T> {
  return request
}

export interface ProtocolInput {
  prepare?: <T extends NativeRequest>(request: T) => Promise<T>
  onPrepared?: (request: ChatCompletionRequestStreaming) => void
  open?: <T>(run: () => Promise<AsyncIterable<T>>) => Promise<AsyncIterable<T>>
  engine: MLCEngineInterface
  modelId: string
  thinkingPrefilled?: boolean
  history: ProtocolMessage[]
  tools: ToolSpec[]
  choice: ToolChoice | 'none' | undefined
  request: WebLLMRequest
  onChunk?: ChunkCallback<WebLLMChunk> | undefined
  /** Defaults to true; adapters can disable snapshots for internal observers. */
  onChunkMayMutate?: boolean | undefined
  signal?: AbortSignal | undefined
}

export function protocolStream({
  engine,
  open = (run) => run(),
  prepare = keepRequest,
  onPrepared,
  modelId,
  thinkingPrefilled = false,
  history,
  tools,
  choice,
  request,
  onChunk,
  onChunkMayMutate = true,
  signal,
}: ProtocolInput): AsyncIterable<StreamChunk> {
  const gemma = isGemma(modelId)
  const preserve = preserveThinking(modelId, request)
  const structured = !isTextFormat(request.response_format)
  if (choice === 'none') tools = []
  validateToolNames(tools)
  const toolResponseFormat = qwenToolResponseFormat(
    modelId,
    request,
    tools,
    choice,
    thinkingPrefilled,
  )
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
    stream_options: streamOptions(request),
    n: 1,
  }
  if (gemmaProtocol) {
    const { createRequest, prefix } = gemmaRequest(history, tools, choice, sampling)
    let stopRequested = Boolean(request.stop?.length)
    return stream(
      async () => {
        const prepared = await prepare(createRequest())
        checkGemmaRequest(prepared)
        if (prefix && !prepared.prompt.endsWith(prefix))
          throw new Error('onPayload cannot change the Gemma protocol prefix.')
        stopRequested = Boolean(prepared.stop?.length)
        signal?.throwIfAborted()
        return engine.completions.create(prepared)
      },
      (chunks) =>
        gemmaChunks(chunks, tools, {
          onChunk,
          onChunkMayMutate,
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
    let parseProtocol = !structured || Boolean(toolResponseFormat)
    let thinkingDisabled = request.extra_body?.enable_thinking === false
    let prefilled = thinkingPrefilled && !thinkingDisabled
    let stopRequested = Boolean(request.stop?.length)
    return stream(
      async () => {
        const prepared = await prepare({
          ...sampling,
          ...(toolResponseFormat && {
            // Keep the generated format intact when a payload hook edits its copy.
            response_format:
              prepare === keepRequest ? toolResponseFormat : structuredClone(toolResponseFormat),
          }),
          messages,
        })
        // The payload hook may change how native output is framed.
        parseProtocol =
          isTextFormat(prepared.response_format) ||
          (toolResponseFormat !== undefined &&
            (prepare === keepRequest || sameJSON(prepared.response_format, toolResponseFormat)))
        thinkingDisabled = prepared.extra_body?.enable_thinking === false
        prefilled = thinkingPrefilled && !thinkingDisabled
        stopRequested = Boolean(prepared.stop?.length)
        if (tools.length && !parseProtocol)
          throw new Error('Qwen tool calls cannot use a structured response format.')
        signal?.throwIfAborted()
        return engine.chat.completions.create(prepared)
      },
      (chunks) =>
        qwenModelChunks(modelId, chunks, tools, {
          onChunk,
          onChunkMayMutate,
          parseProtocol,
          thinkingPrefilled: prefilled,
          thinkingDisabled,
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
  let hasTools = false
  return stream(
    async () => {
      // Hermes adds its system prompt and response format while opening a request.
      const prepared = await prepare({ ...native, messages: [...native.messages] })
      onPrepared?.(prepared)
      // WebLLM treats an empty tools list as native tool mode, too.
      hasTools = Boolean(prepared.tools)
      signal?.throwIfAborted()
      return engine.chat.completions.create(prepared)
    },
    (chunks) => nativeChunks(chunks, hasTools, { onChunk, onChunkMayMutate, signal }),
  )

  function stream<T>(
    create: () => Promise<AsyncIterable<T>>,
    read: (chunks: AsyncIterable<T>) => AsyncIterable<StreamChunk>,
  ) {
    return streamCall(engine, () => open(create), read, signal)
  }
}
