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
import type { BrowserPage, BrowserRequest } from './types.js'

/** Runs the adapter in a controlled browser; Harness and its tools stay in Node.js. */
export class BrowserModel extends Model<WebLLMModelConfig> {
  private readonly page: BrowserPage
  private config: WebLLMModelConfig

  constructor(page: BrowserPage, config: WebLLMModelConfig) {
    super()
    this.page = page
    this.config = structuredClone(config)
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
      JSON.stringify({
        messages: messages.map((message) => message.toJSON()),
        config: this.getConfig(),
        options: { ...options, systemPrompt: formatSystemPrompt(options.systemPrompt) },
      }),
    )
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const { cancelSignal, systemPrompt, toolSpecs, toolChoice, dynamicTrailingBlocks } = options
    cancelSignal?.throwIfAborted()
    const prompt = formatSystemPrompt(systemPrompt)
    const request: BrowserRequest = {
      messages: messages.map((message) => message.toJSON()),
      config: this.getConfig(),
      options: {
        ...(prompt !== undefined && { systemPrompt: prompt }),
        ...(toolSpecs !== undefined && { toolSpecs }),
        ...(toolChoice !== undefined && { toolChoice }),
        ...(dynamicTrailingBlocks !== undefined && { dynamicTrailingBlocks }),
      },
    }
    // SDK message JSON encodes media bytes as base64 across the browser boundary.
    const id = await this.page.evaluate(
      (json) => window.webllmBridge.start(JSON.parse(json)),
      JSON.stringify(request),
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
