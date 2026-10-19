import type { BaseMessage } from '@langchain/core/messages'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { ChatGenerationChunk } from '@langchain/core/outputs'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { WebLLMChatModel } from './chat/base.js'
import { formatMessages } from './chat/messages.js'
import { formatChoice, formatTools } from './chat/tools.js'
import { protocolStream } from './protocol/stream.js'
import { readChunks } from './chat/stream.js'
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
    const retries = options.maxRetries ?? this.retries
    const chunks = protocolStream({
      engine: this.engine,
      caller: options.signal && retries > 0 ? this.createCaller() : this.caller,
      modelId: this.modelId,
      history: formatMessages(messages),
      tools: formatTools(options.tools),
      choice,
      request,
      onChunk: this.onChunk,
      signal: options.signal,
      maxRetries: options.maxRetries,
    })
    for await (const chunk of readChunks(chunks, this.modelId, options.signal, choice)) {
      if (runManager)
        await runManager.handleLLMNewToken(chunk.text, undefined, undefined, undefined, undefined, {
          chunk,
        })
      yield chunk
    }
  }
}
