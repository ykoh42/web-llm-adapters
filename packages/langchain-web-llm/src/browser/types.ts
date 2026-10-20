import type { ErrorInfo } from '../../../../shared/web-llm/errors.js'
import type { BrowserPacket } from '../../../../shared/browser/stream.js'
import type { BrowserError } from './errors.js'
import type { FailedAttempt } from './failed-attempts.js'
import type { WebLLMEmbeddingsOptions } from '../embeddings/model.js'
import type {
  AIMessageChunkFields,
  StandardMessageStructure,
  MessageFieldWithRole,
} from '@langchain/core/messages'
import type { WebLLMModelOptions, WebLLMCallOptions, WebLLMChunk } from '../types.js'

/** A trusted browser page controlled by the application, such as a Playwright Page. */
export interface BrowserPage {
  evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T>
}
export interface BrowserRequest {
  includeChunks?: boolean
  includeFailedAttempts?: boolean
  config: WebLLMModelOptions
  messages: MessageFieldWithRole[]
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
  BrowserPacket<BrowserChunk, WebLLMChunk> | FailedAttempt | { error: BrowserError }
) & {
  chunks?: WebLLMChunk[]
}

/** Expose only on a private page controlled by your application. */
export interface BrowserBridge {
  embed(request: {
    config: Omit<WebLLMEmbeddingsOptions, 'engine'>
    texts: string[]
    type: 'query' | 'document'
  }): Promise<number[][] | { error: ErrorInfo }>
  start(request: BrowserRequest): Promise<string>
  /** Returns model output or a native chunk batch. */
  next(id: string): Promise<BrowserResult>
  /** Resume opening the stream after the caller's failed-attempt handler returns. */
  retry(id: string, error: BrowserError): void
  stop(id: string): Promise<void>
  close(): Promise<void>
}

declare global {
  interface Window {
    webllmLangChainBridge?: BrowserBridge
  }
}
