import {
  Model,
  ContextWindowOverflowError,
  type CountTokensOptions,
  type Message,
  type ModelStreamEvent,
  type StreamOptions,
  type SystemPrompt,
} from '@strands-agents/sdk'
import type { WebLLMModelConfig } from 'strands-webllm'
import type { Page } from '@playwright/test'
import type { HarnessRequest } from '../browser/src/harness-api.js'

/** Runs the adapter in a controlled browser; Harness and its tools stay in Node.js. */
export class BrowserModel extends Model<WebLLMModelConfig> {
  private readonly page: Page
  private config: WebLLMModelConfig

  constructor(page: Page, config: WebLLMModelConfig) {
    super()
    this.page = page
    this.config = structuredClone({ contextWindowLimit: 8192, ...config })
  }

  override getConfig() {
    return structuredClone(this.config)
  }

  override updateConfig(config: Partial<WebLLMModelConfig>) {
    this.config = structuredClone({ ...this.config, ...config })
  }

  override async countTokens(messages: Message[], options: CountTokensOptions = {}) {
    return this.page.evaluate(
      (json) => window.harness.count(JSON.parse(json)),
      JSON.stringify({
        messages: messages.map((message) => message.toJSON()),
        config: this.getConfig(),
        options: { ...options, systemPrompt: systemText(options.systemPrompt) },
      }),
    )
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const { cancelSignal, systemPrompt, toolSpecs, toolChoice, dynamicTrailingBlocks } = options
    cancelSignal?.throwIfAborted()
    const prompt = systemText(systemPrompt)
    const request: HarnessRequest = {
      messages: messages.map((message) => message.toJSON()),
      config: this.getConfig(),
      options: {
        ...(prompt !== undefined && { systemPrompt: prompt }),
        ...(toolSpecs !== undefined && { toolSpecs }),
        ...(toolChoice !== undefined && { toolChoice }),
        ...(dynamicTrailingBlocks !== undefined && { dynamicTrailingBlocks }),
      },
    }
    // This carries text, reasoning, and tool JSON. Media stays in the browser example.
    const id = await this.page.evaluate(
      (json) => window.harness.start(JSON.parse(json)),
      JSON.stringify(request),
    )
    // Strands may leave its iterator suspended on cancel, so stop the browser directly.
    const stop = () => {
      void this.page.evaluate((id) => window.harness.stop(id), id).catch(() => {})
    }
    cancelSignal?.addEventListener('abort', stop, { once: true })
    try {
      cancelSignal?.throwIfAborted()
      while (true) {
        const result = await this.page.evaluate((id) => window.harness.next(id), id)
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
      await this.page.evaluate((id) => window.harness.stop(id), id)
    }
  }
}

function systemText(prompt?: SystemPrompt) {
  if (typeof prompt === 'string' || !prompt) return prompt
  return prompt
    .filter((block) => block.type === 'textBlock')
    .map((block) => block.text)
    .join('\n')
}
