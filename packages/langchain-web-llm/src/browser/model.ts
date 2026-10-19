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
    const { tool_choice, request, stop, response_format, maxRetries } = options
    const input: BrowserRequest = {
      ...(this.onChunk && { includeChunks: true }),
      config: { modelId: this.modelId, request: this.request, maxRetries: this.retries },
      messages: messages.map((message) => message.toDict()),
      options: {
        ...(options.tools && { tools: options.tools.map((tool) => convertToOpenAITool(tool)) }),
        ...(tool_choice !== undefined && { tool_choice }),
        ...(request && { request }),
        ...(stop && { stop }),
        ...(response_format && { response_format }),
        ...(maxRetries !== undefined && { maxRetries }),
      },
    }
    const id = await this.page.evaluate((request) => {
      const bridge = window.webllmLangChainBridge
      if (!bridge)
        throw new Error(
          'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserChatWebLLM.',
        )
      return bridge.start(request)
    }, input)
    let closing: Promise<void> | undefined
    const close = () =>
      (closing ??= this.page.evaluate((id) => {
        const bridge = window.webllmLangChainBridge
        if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
        return bridge.stop(id)
      }, id))
    const cancel = () => {
      void close().catch(() => {})
    }
    signal?.addEventListener('abort', cancel, { once: true })
    let failed = false
    let complete = false
    try {
      signal?.throwIfAborted()
      while (true) {
        const result = await this.page.evaluate((id) => {
          const bridge = window.webllmLangChainBridge
          if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
          return bridge.next(id)
        }, id)
        signal?.throwIfAborted()
        for (const chunk of result.chunks ?? []) this.onChunk?.(chunk)
        if ('error' in result) {
          complete = true
          throw Object.assign(new Error(result.error.message), { name: result.error.name })
        }
        if (result.done) {
          complete = true
          return
        }
        const chunk = new ChatGenerationChunk({
          ...result.value,
          message: new AIMessageChunk<StandardMessageStructure>(result.value.message),
        })
        if (runManager)
          await runManager.handleLLMNewToken(
            chunk.text,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk },
          )
        yield chunk
      }
    } catch (error) {
      failed = true
      throw error
    } finally {
      signal?.removeEventListener('abort', cancel)
      if (!complete) {
        if (failed) await close().catch(() => {})
        else await close()
      }
    }
  }
}
