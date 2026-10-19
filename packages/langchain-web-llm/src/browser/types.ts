import type { WebLLMEmbeddingsOptions } from '../embeddings/model.js'
import type {
  AIMessageChunkFields,
  StandardMessageStructure,
  StoredMessage,
} from '@langchain/core/messages'
import type { WebLLMModelOptions, WebLLMCallOptions, WebLLMChunk } from '../types.js'

/** A trusted browser page controlled by the application, such as a Playwright Page. */
export interface BrowserPage {
  evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T>
}
export interface BrowserRequest {
  includeChunks?: boolean
  config: WebLLMModelOptions
  messages: StoredMessage[]
  options: Pick<
    WebLLMCallOptions,
    'tools' | 'tool_choice' | 'request' | 'stop' | 'response_format' | 'maxRetries'
  >
}
export interface BrowserChunk {
  text: string
  message: AIMessageChunkFields<StandardMessageStructure>
  generationInfo?: Record<string, unknown>
}
type BrowserResult = (
  IteratorResult<BrowserChunk, undefined> | { error: { name: string; message: string } }
) & { chunks?: WebLLMChunk[] }

/** Expose only on a private page controlled by your application. */
export interface BrowserBridge {
  embed(request: {
    config: Omit<WebLLMEmbeddingsOptions, 'engine'>
    texts: string[]
    type: 'query' | 'document'
  }): Promise<number[][]>
  start(request: BrowserRequest): Promise<string>
  next(id: string): Promise<BrowserResult>
  stop(id: string): Promise<void>
  close(): Promise<void>
}

declare global {
  interface Window {
    webllmLangChainBridge?: BrowserBridge
  }
}
