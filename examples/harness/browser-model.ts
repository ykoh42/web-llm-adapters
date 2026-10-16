import { Model, type Message, type ModelStreamEvent, type StreamOptions } from '@strands-agents/sdk'
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
    this.config = structuredClone(config)
  }

  override getConfig() {
    return structuredClone(this.config)
  }

  override updateConfig(config: Partial<WebLLMModelConfig>) {
    this.config = structuredClone({ ...this.config, ...config })
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const { cancelSignal, systemPrompt, toolSpecs, toolChoice } = options
    cancelSignal?.throwIfAborted()
    const prompt =
      typeof systemPrompt === 'string'
        ? systemPrompt
        : systemPrompt
            ?.filter((block) => block.type === 'textBlock')
            .map((block) => block.text)
            .join('\n')
    const request: HarnessRequest = {
      messages: messages.map((message) => message.toJSON()),
      config: this.getConfig(),
      options: {
        ...(prompt !== undefined && { systemPrompt: prompt }),
        ...(toolSpecs !== undefined && { toolSpecs }),
        ...(toolChoice !== undefined && { toolChoice }),
      },
    }
    // This carries text, reasoning, and tool JSON. Media stays in the browser example.
    await this.page.evaluate(
      (json) => window.harness.start(JSON.parse(json)),
      JSON.stringify(request),
    )
    // Strands may leave its iterator suspended on cancel, so stop the browser directly.
    const stop = () => {
      void this.page.evaluate(() => window.harness.stop()).catch(() => {})
    }
    cancelSignal?.addEventListener('abort', stop, { once: true })
    try {
      cancelSignal?.throwIfAborted()
      while (true) {
        const result = await this.page.evaluate(() => window.harness.next())
        if (result.done) return
        yield result.value
      }
    } finally {
      cancelSignal?.removeEventListener('abort', stop)
      await this.page.evaluate(() => window.harness.stop())
    }
  }
}
