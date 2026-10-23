import type { ErrorInfo } from '../../../../shared/web-llm/errors.js'
import type { BrowserPacket } from '../../../../shared/browser/stream.js'
import type { MessageData, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk'
import type { WebLLMChunk, WebLLMModelConfig, WebLLMModelOptions } from '../types.js'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'

export type BrowserModelOptions = Omit<WebLLMModelOptions, 'engine'>

export interface BrowserRequest {
  includeChunks?: boolean
  messages: MessageData[]
  config: WebLLMModelConfig
  options: Pick<StreamOptions, 'toolSpecs' | 'toolChoice' | 'dynamicTrailingBlocks'> & {
    systemPrompt?: string
  }
}
export type BrowserError = ErrorInfo
export type BrowserResult = (
  | BrowserPacket<ModelStreamEvent, WebLLMChunk>
  | { error: BrowserError; readonly errorJSON?: string }
) & {
  chunks?: WebLLMChunk[]
}

/** A private page controlled by the caller, such as a Playwright Page. */
export interface BrowserPage {
  evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T>
}

/** Expose this object only on a trusted page controlled by your application. */
export interface BrowserBridge {
  /** Run on the browser page. Finish native work inside the callback before returning. */
  withEngine<T>(run: (engine: MLCEngineInterface) => T | Promise<T>): Promise<T>
  start(request: BrowserRequest): Promise<string>
  /** Returns model output or a native chunk batch. */
  next(id: string): Promise<BrowserResult>
  stop(id: string): Promise<void>
  close(): Promise<void>
}

declare global {
  interface Window {
    webllmBridge: BrowserBridge
  }
}
