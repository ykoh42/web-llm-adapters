import {
  AIMessageChunk,
  type BaseMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import { callChunkCallback, waitForCallback } from '../../../../shared/web-llm/callback.js'
import { restoreBrowserError, serializeBrowserError } from './errors.js'
import { restoreNegativeZeros } from './tool-calls.js'
import { browserMessages } from './messages.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import { startBrowserCall } from '../../../../shared/browser/start.js'
import { watchBrowserCall } from '../../../../shared/browser/lifecycle.js'
import { readBrowserError } from '../../../../shared/browser/errors.js'
import { WebLLMChatModel } from '../chat/base.js'
import { hasTokenCallbacks, notifyToken } from '../chat/callbacks.js'
import { modelError } from '../error.js'
import type { WebLLMModelOptions, ChatWebLLMOptions } from '../types.js'
import type { BrowserChunk, BrowserPage, BrowserRequest } from './types.js'
import type { FailedAttempt } from './failed-attempts.js'

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
    return this.finishResult(result)
  }

  private async *readResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager: CallbackManagerForLLMRun | undefined,
    batchSize: number,
  ): AsyncGenerator<ChatGenerationChunk> {
    const signal = options.signal
    signal?.throwIfAborted()
    const input = this.createRequest(messages, options)
    const stopCall = (id: string) =>
      this.page.evaluate((id) => {
        const bridge = window.webllmLangChainBridge
        if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
        return bridge.stop(id)
      }, id)
    // Encode model inputs so browser transports keep reserved JSON keys intact.
    const start = this.page.evaluate((input) => {
      const request: BrowserRequest = JSON.parse(input)
      const bridge = window.webllmLangChainBridge
      if (!bridge)
        throw new Error(
          'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserChatWebLLM.',
        )
      return bridge.start(request)
    }, stringifyJSON(input))
    const id = await startBrowserCall(start, stopCall, signal)
    const finish = watchBrowserCall(id, stopCall, signal)
    let failed = false
    let complete = false
    try {
      signal?.throwIfAborted()
      while (true) {
        const results = await waitForCallback(
          this.page.evaluate(readBrowserBatch, {
            id,
            batchSize: hasTokenCallbacks(runManager) ? 1 : batchSize,
          }),
          signal,
        )
        for (const result of results) {
          for (const chunk of result.chunks ?? []) {
            signal?.throwIfAborted()
            const pending = callChunkCallback(this.onChunk, chunk, signal)
            if (pending) await pending
          }
          signal?.throwIfAborted()
          if ('error' in result) {
            complete = true
            throw restoreBrowserError(readBrowserError(result))
          }
          if ('failedAttempt' in result) {
            await this.retryCall(id, result, signal)
            break
          }
          if (result.done) {
            complete = true
            return
          }
          if ('native' in result) continue
          const chunk = readBrowserChunk(result.value)
          if (runManager && hasTokenCallbacks(runManager))
            await waitForCallback(notifyToken(runManager, chunk), signal)
          signal?.throwIfAborted()
          yield chunk
        }
      }
    } catch (error) {
      failed = true
      if (signal?.aborted) throw error
      throw modelError(error)
    } finally {
      const pending = finish(complete, failed)
      if (pending) await pending
    }
  }

  private createRequest(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
  ): BrowserRequest {
    const { tool_choice, maxRetries } = options
    return {
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
  }

  private async retryCall(id: string, result: FailedAttempt, signal?: AbortSignal): Promise<void> {
    const error = restoreBrowserError(
      result.failedAttemptJSON === undefined
        ? result.failedAttempt
        : JSON.parse(result.failedAttemptJSON),
    )
    await waitForCallback(
      Promise.resolve().then(() => {
        signal?.throwIfAborted()
        return this.onFailedAttempt!(error)
      }),
      signal,
    )
    signal?.throwIfAborted()
    await waitForCallback(
      this.page.evaluate(
        ({ id, error }) => {
          const bridge = window.webllmLangChainBridge
          if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
          bridge.retry(id, JSON.parse(error))
        },
        { id, error: JSON.stringify(serializeBrowserError(error)) },
      ),
      signal,
    )
  }
}

async function readBrowserBatch({ id, batchSize }: { id: string; batchSize: number }) {
  const bridge = window.webllmLangChainBridge
  if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
  const results = []
  for (let index = 0; index < batchSize; index++) {
    const result = await bridge.next(id)
    results.push(result)
    if (('done' in result && result.done) || 'error' in result || 'failedAttempt' in result) break
  }
  return results
}

function readBrowserChunk(value: BrowserChunk): ChatGenerationChunk {
  const message = new AIMessageChunk<StandardMessageStructure>(value.message)
  const toolCalls = value.toolCallsJSON ? JSON.parse(value.toolCallsJSON) : value.message.tool_calls
  if (toolCalls && value.toolCallNegativeZeros)
    restoreNegativeZeros(toolCalls, value.toolCallNegativeZeros)
  if (toolCalls?.length) {
    message.tool_calls = toolCalls
    message.lc_kwargs.tool_calls = message.tool_calls
  }
  return new ChatGenerationChunk({ ...value, message })
}
