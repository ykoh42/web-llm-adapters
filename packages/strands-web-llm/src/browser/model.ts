import type { Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk'
import { fromBrowserError } from './errors.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import { WebLLMModelBase } from '../model-base.js'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../chat/messages.js'
import type { BrowserModelOptions, BrowserPage, BrowserRequest } from './types.js'

/** Runs the adapter in a controlled browser; Harness and its tools stay in Node.js. */
export class BrowserModel extends WebLLMModelBase {
  private readonly page: BrowserPage
  private readonly onChunk: BrowserModelOptions['onChunk']

  constructor(page: BrowserPage, { onChunk, ...config }: BrowserModelOptions) {
    super(config)
    this.page = page
    this.onChunk = onChunk
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const { cancelSignal } = options
    cancelSignal?.throwIfAborted()
    const id = await this.page.evaluate(
      (input) => {
        const request: BrowserRequest = JSON.parse(input)
        const bridge = window.webllmBridge
        if (!bridge)
          throw new Error(
            'Expose createBrowserBridge(engine) as window.webllmBridge before using BrowserModel.',
          )
        return bridge.start(request)
      },
      encodeRequest(messages, this.getConfig(), options, Boolean(this.onChunk)),
    )
    // Strands may leave its iterator suspended on cancel, so stop the browser directly.
    let closing: Promise<void> | undefined
    const close = () => (closing ??= this.page.evaluate((id) => window.webllmBridge.stop(id), id))
    const stop = () => {
      void close().catch(() => {})
    }
    cancelSignal?.addEventListener('abort', stop, { once: true })
    let failed = false
    let complete = false
    try {
      cancelSignal?.throwIfAborted()
      while (true) {
        const result = await this.page.evaluate((id) => window.webllmBridge.next(id), id)
        for (const chunk of result.chunks ?? []) {
          cancelSignal?.throwIfAborted()
          this.onChunk?.(chunk)
        }
        cancelSignal?.throwIfAborted()
        if ('error' in result) {
          complete = true
          throw fromBrowserError(result.error)
        }
        if (result.done) {
          complete = true
          return
        }
        if ('native' in result) continue
        yield result.value
      }
    } catch (error) {
      failed = true
      throw error
    } finally {
      cancelSignal?.removeEventListener('abort', stop)
      if (!complete) {
        if (failed) await close().catch(() => {})
        else await close()
      }
    }
  }
}

/** Send only provider inputs; SDK message JSON preserves media bytes as base64. */
function encodeRequest(
  messages: Message[],
  config: WebLLMModelConfig,
  options: StreamOptions,
  includeChunks = false,
) {
  const { systemPrompt, toolSpecs, toolChoice, dynamicTrailingBlocks } = options
  const prompt = formatSystemPrompt(systemPrompt)
  const request: BrowserRequest = {
    ...(includeChunks && { includeChunks }),
    messages: messages.map((message) => {
      const { role, content } = message.toJSON()
      return { role, content }
    }),
    config,
    options: {
      ...(prompt !== undefined && { systemPrompt: prompt }),
      ...(toolSpecs !== undefined && { toolSpecs }),
      ...(toolChoice !== undefined && { toolChoice }),
      ...(dynamicTrailingBlocks !== undefined && { dynamicTrailingBlocks }),
    },
  }
  // Browser transports may discard reserved object keys; JSON preserves their literal values.
  return stringifyJSON(request)
}
