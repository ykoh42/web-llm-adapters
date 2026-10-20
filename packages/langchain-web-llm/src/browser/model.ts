import {
  AIMessageChunk,
  type BaseMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import { waitForCallback } from './failed-attempts.js'
import { restoreBrowserError, serializeBrowserError } from './errors.js'
import { browserMessages } from './messages.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import { WebLLMChatModel } from '../chat/base.js'
import type { WebLLMModelOptions, ChatWebLLMOptions } from '../types.js'
import type { BrowserPage, BrowserRequest } from './types.js'

/** Keep the agent and tools in Node.js while WebLLM runs in a caller-controlled browser. */
export class BrowserChatWebLLM extends WebLLMChatModel {
  private readonly onChunk: ChatWebLLMOptions['onChunk']
  private readonly onFailedAttempt: WebLLMModelOptions['onFailedAttempt']
  constructor(
    private readonly page: BrowserPage,
    { onChunk, ...options }: WebLLMModelOptions & Pick<ChatWebLLMOptions, 'onChunk'>,
  ) {
    super(options)
    this.onChunk = onChunk
    this.onFailedAttempt = options.onFailedAttempt
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const signal = options.signal
    signal?.throwIfAborted()
    const { tool_choice, maxRetries } = options
    const input: BrowserRequest = {
      ...(this.onChunk && { includeChunks: true }),
      ...(this.onFailedAttempt && { includeFailedAttempts: true }),
      config: {
        modelId: this.modelId,
        request: this.formatRequest(options),
        maxRetries: this.retries,
      },
      messages: browserMessages(messages),
      options: {
        ...(options.tools && { tools: options.tools.map((tool) => convertToOpenAITool(tool)) }),
        ...(tool_choice !== undefined && { tool_choice }),
        ...(maxRetries !== undefined && { maxRetries }),
      },
    }
    // Encode model inputs so browser transports keep reserved JSON keys intact.
    const id = await this.page.evaluate((input) => {
      const request: BrowserRequest = JSON.parse(input)
      const bridge = window.webllmLangChainBridge
      if (!bridge)
        throw new Error(
          'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserChatWebLLM.',
        )
      return bridge.start(request)
    }, stringifyJSON(input))
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
        for (const chunk of result.chunks ?? []) {
          signal?.throwIfAborted()
          this.onChunk?.(chunk)
        }
        signal?.throwIfAborted()
        if ('error' in result) {
          complete = true
          throw restoreBrowserError(result.error)
        }
        if ('failedAttempt' in result) {
          const error = restoreBrowserError(result.failedAttempt)
          await waitForCallback(
            Promise.resolve().then(() => this.onFailedAttempt!(error)),
            signal,
          )
          signal?.throwIfAborted()
          await this.page.evaluate(
            ({ id, error }) => {
              const bridge = window.webllmLangChainBridge
              if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
              bridge.retry(id, error)
            },
            { id, error: serializeBrowserError(error) },
          )
          continue
        }
        if (result.done) {
          complete = true
          return
        }
        if ('native' in result) continue
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
        signal?.throwIfAborted()
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
