import {
  Model,
  ContextWindowOverflowError,
  type CountTokensOptions,
  type Message,
  type ModelStreamEvent,
  type StreamOptions,
} from '@strands-agents/sdk'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../messages.js'
import type { BrowserModelOptions, BrowserPage, BrowserRequest } from './types.js'

/** Runs the adapter in a controlled browser; Harness and its tools stay in Node.js. */
export class BrowserModel extends Model<WebLLMModelConfig> {
  private readonly page: BrowserPage
  private config: WebLLMModelConfig
  private readonly onChunk: BrowserModelOptions['onChunk']

  constructor(page: BrowserPage, { onChunk, ...config }: BrowserModelOptions) {
    super()
    this.page = page
    this.config = structuredClone(config)
    this.onChunk = onChunk
  }

  override getConfig() {
    return structuredClone(this.config)
  }

  override updateConfig(config: Partial<WebLLMModelConfig>) {
    this.config = structuredClone({ ...this.config, ...config })
  }

  override async countTokens(messages: Message[], options: CountTokensOptions = {}) {
    return this.page.evaluate(
      (json) => window.webllmBridge.count(JSON.parse(json)),
      encodeRequest(messages, this.getConfig(), options),
    )
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const { cancelSignal } = options
    cancelSignal?.throwIfAborted()
    const id = await this.page.evaluate(
      (json) => window.webllmBridge.start(JSON.parse(json)),
      encodeRequest(messages, this.getConfig(), options, Boolean(this.onChunk)),
    )
    // Strands may leave its iterator suspended on cancel, so stop the browser directly.
    let closing: Promise<void> | undefined
    const close = () => (closing ??= this.page.evaluate((id) => window.webllmBridge.stop(id), id))
    const stop = () => {
      void close().catch(() => {})
    }
    cancelSignal?.addEventListener('abort', stop, { once: true })
    try {
      cancelSignal?.throwIfAborted()
      while (true) {
        const result = await this.page.evaluate((id) => window.webllmBridge.next(id), id)
        cancelSignal?.throwIfAborted()
        for (const chunk of result.chunks ?? []) this.onChunk?.(chunk)
        if ('error' in result) {
          const { name, message } = result.error
          if (name === 'ContextWindowOverflowError') throw new ContextWindowOverflowError(message)
          throw Object.assign(new Error(message), { name })
        }
        if (result.done) return
        yield result.value
      }
    } finally {
      cancelSignal?.removeEventListener('abort', stop)
      await close()
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
    messages: messages.map((message) => message.toJSON()),
    config,
    options: {
      ...(prompt !== undefined && { systemPrompt: prompt }),
      ...(toolSpecs !== undefined && { toolSpecs }),
      ...(toolChoice !== undefined && { toolChoice }),
      ...(dynamicTrailingBlocks !== undefined && { dynamicTrailingBlocks }),
    },
  }
  return JSON.stringify(request)
}
