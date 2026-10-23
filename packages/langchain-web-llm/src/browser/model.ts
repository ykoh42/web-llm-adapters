import {
  AIMessageChunk,
  type BaseMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import { waitForCallback } from '../../../../shared/web-llm/callback.js'
import { restoreBrowserError, serializeBrowserError } from './errors.js'
import { browserMessages } from './messages.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import { WebLLMChatModel } from '../chat/base.js'
import { hasTokenCallbacks } from '../chat/callbacks.js'
import { modelError } from '../error.js'
import type { WebLLMModelOptions, ChatWebLLMOptions } from '../types.js'
import type { BrowserPage, BrowserRequest } from './types.js'

// invoke buffers its result; streaming and per-chunk callbacks still read one packet at a time.
const bufferedReadSize = 512

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
    yield* this.readResponseChunks(messages, options, runManager, 1)
  }

  override async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const needsSingleChunkReads = Boolean(hasTokenCallbacks(runManager) || this.onChunk)
    const batchSize = needsSingleChunkReads ? 1 : bufferedReadSize
    let result: ChatGenerationChunk | undefined
    for await (const chunk of this.readResponseChunks(messages, options, runManager, batchSize))
      result = result ? result.concat(chunk) : chunk
    if (!result) throw new Error('WebLLM returned no response.')
    return { generations: [result] }
  }

  private async *readResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager: CallbackManagerForLLMRun | undefined,
    batchSize: number,
  ): AsyncGenerator<ChatGenerationChunk> {
    const signal = options.signal
    const notifyTokenCallbacks = hasTokenCallbacks(runManager)
    signal?.throwIfAborted()
    const { tool_choice, maxRetries } = options
    const input: BrowserRequest = {
      ...(this.onChunk && { includeChunks: true }),
      ...(this.onFailedAttempt && { includeFailedAttempts: true }),
      config: {
        modelId: this.modelId,
        ...(this.thinkingPrefilled && { thinkingPrefilled: true }),
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
        const results = await this.page.evaluate(
          async ({ id, batchSize }) => {
            const bridge = window.webllmLangChainBridge
            if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
            const results = []
            for (let index = 0; index < batchSize; index++) {
              const result = await bridge.next(id)
              results.push(result)
              if (
                ('done' in result && result.done) ||
                'error' in result ||
                'failedAttempt' in result
              )
                break
            }
            return results
          },
          { id, batchSize },
        )
        for (const result of results) {
          for (const chunk of result.chunks ?? []) {
            signal?.throwIfAborted()
            this.onChunk?.(chunk)
          }
          signal?.throwIfAborted()
          if ('error' in result) {
            complete = true
            throw restoreBrowserError(
              result.errorJSON === undefined ? result.error : JSON.parse(result.errorJSON),
            )
          }
          if ('failedAttempt' in result) {
            const error = restoreBrowserError(
              result.failedAttemptJSON === undefined
                ? result.failedAttempt
                : JSON.parse(result.failedAttemptJSON),
            )
            await waitForCallback(
              Promise.resolve().then(() => this.onFailedAttempt!(error)),
              signal,
            )
            signal?.throwIfAborted()
            await this.page.evaluate(
              ({ id, error }) => {
                const bridge = window.webllmLangChainBridge
                if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
                bridge.retry(id, JSON.parse(error))
              },
              { id, error: JSON.stringify(serializeBrowserError(error)) },
            )
            break
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
          if (runManager && notifyTokenCallbacks)
            await waitForCallback(
              runManager.handleLLMNewToken(chunk.text, undefined, undefined, undefined, undefined, {
                chunk,
              }),
              signal,
            )
          signal?.throwIfAborted()
          yield chunk
        }
      }
    } catch (error) {
      failed = true
      if (signal?.aborted) throw error
      throw modelError(error)
    } finally {
      signal?.removeEventListener('abort', cancel)
      if (!complete) {
        if (failed) await close().catch(() => {})
        else await close()
      }
    }
  }
}
