import {
  AIMessageChunk,
  type BaseMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import { WebLLMChatModel } from '../chat/base.js'
import type { WebLLMModelOptions, ChatWebLLMOptions } from '../types.js'
import type { BrowserPage, BrowserRequest } from './types.js'

/** Keep the agent and tools in Node.js while WebLLM runs in a caller-controlled browser. */
export class BrowserChatWebLLM extends WebLLMChatModel {
  private readonly onChunk: ChatWebLLMOptions['onChunk']
  constructor(
    private readonly page: BrowserPage,
    { onChunk, ...options }: WebLLMModelOptions & Pick<ChatWebLLMOptions, 'onChunk'>,
  ) {
    super(options)
    this.onChunk = onChunk
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const signal = options.signal
    signal?.throwIfAborted()
    const { tool_choice, request, stop } = options
    const input: BrowserRequest = {
      ...(this.onChunk && { includeChunks: true }),
      config: { modelId: this.modelId, request: this.request, maxRetries: this.retries },
      messages: messages.map((message) => message.toDict()),
      options: {
        ...(options.tools && { tools: options.tools.map((tool) => convertToOpenAITool(tool)) }),
        ...(tool_choice !== undefined && { tool_choice }),
        ...(request && { request }),
        ...(stop && { stop }),
      },
    }
    const id = await this.page.evaluate(
      (json) => window.webllmLangChainBridge.start(JSON.parse(json)),
      JSON.stringify(input),
    )
    let closing: Promise<void> | undefined
    const close = () =>
      (closing ??= this.page.evaluate((id) => window.webllmLangChainBridge.stop(id), id))
    const cancel = () => {
      void close().catch(() => {})
    }
    signal?.addEventListener('abort', cancel, { once: true })
    try {
      signal?.throwIfAborted()
      while (true) {
        const result = await this.page.evaluate((id) => window.webllmLangChainBridge.next(id), id)
        signal?.throwIfAborted()
        for (const chunk of result.chunks ?? []) this.onChunk?.(chunk)
        if ('error' in result)
          throw Object.assign(new Error(result.error.message), { name: result.error.name })
        if (result.done) return
        const chunk = new ChatGenerationChunk({
          ...result.value,
          message: new AIMessageChunk<StandardMessageStructure>(result.value.message),
        })
        yield chunk
        if (chunk.text)
          await runManager?.handleLLMNewToken(
            chunk.text,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk },
          )
      }
    } finally {
      signal?.removeEventListener('abort', cancel)
      await close()
    }
  }
}
