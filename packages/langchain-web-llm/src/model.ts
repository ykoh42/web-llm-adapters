import type { BaseMessage } from '@langchain/core/messages'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { WebLLMChatModel } from './chat/base.js'
import { hasTokenCallbacks, notifyToken } from './chat/callbacks.js'
import { formatMessages } from './chat/messages.js'
import { formatChoice, formatTools } from './chat/tools.js'
import { protocolStream } from './protocol/stream.js'
import { modelError } from './error.js'
import { readChunks } from './chat/chunks.js'
import { appendLogprobs, type WebLLMLogprobs } from '../../../shared/web-llm/logprobs.js'
import { chunkCallbackMayMutate, waitForCallback } from '../../../shared/web-llm/callback.js'
import type { ChatWebLLMOptions } from './types.js'

/** Use a caller-owned WebLLM engine with LangChain, LangGraph, or Deep Agents. */
export class ChatWebLLM extends WebLLMChatModel {
  readonly engine: MLCEngineInterface
  private readonly onChunk: ChatWebLLMOptions['onChunk']

  constructor({ engine, onChunk, ...params }: ChatWebLLMOptions) {
    super(params)
    this.engine = engine
    this.onChunk = onChunk
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    options.signal?.throwIfAborted()
    const request = this.formatRequest(options)
    const choice = formatChoice(options.tool_choice)
    const history = formatMessages(messages)
    const tools = formatTools(options.tools)
    let responseId: string | undefined
    let responseModel: string | undefined
    let responseCreated: number | undefined
    let logprobs: WebLLMLogprobs | undefined
    const chunks = protocolStream({
      engine: this.engine,
      caller: options.signal ? this.createCaller(options.signal) : this.caller,
      modelId: this.modelId,
      thinkingPrefilled: this.thinkingPrefilled,
      history,
      tools,
      choice,
      request,
      onChunkMayMutate: chunkCallbackMayMutate(this.onChunk),
      onChunk: (chunk) => {
        responseId ??= chunk.id
        responseModel ??= chunk.model
        responseCreated ??= chunk.created
        logprobs = appendLogprobs(logprobs, chunk.choices[0]?.logprobs)
        return this.onChunk?.(chunk)
      },
      signal: options.signal,
      maxRetries: options.maxRetries,
    })
    function addMetadata(chunk: ChatGenerationChunk) {
      if (responseCreated !== undefined) chunk.message.response_metadata.created = responseCreated
      if (responseModel) {
        chunk.message.response_metadata.model = responseModel
        chunk.message.response_metadata.model_name = responseModel
      }
      if (logprobs) {
        chunk.message.response_metadata.logprobs = logprobs
        chunk.generationInfo!.logprobs = logprobs
      }
    }
    try {
      for await (const chunk of readChunks(
        chunks,
        this.modelId,
        options.signal,
        choice,
        tools.map((tool) => tool.name),
      )) {
        options.signal?.throwIfAborted()
        if (responseId) chunk.message.id = responseId
        if (chunk.generationInfo?.finish_reason) addMetadata(chunk)
        if (runManager && hasTokenCallbacks(runManager))
          await waitForCallback(notifyToken(runManager, chunk), options.signal)
        options.signal?.throwIfAborted()
        yield chunk
      }
    } catch (error) {
      if (options.signal?.aborted) throw error
      throw modelError(error)
    }
  }
}
