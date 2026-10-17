import type {
  CountTokensOptions,
  MessageData,
  ModelStreamEvent,
  StreamOptions,
} from '@strands-agents/sdk'
import type { WebLLMChunk, WebLLMModelConfig, WebLLMModelOptions } from '../types.js'

export type BrowserModelOptions = Omit<WebLLMModelOptions, 'engine'>

export interface BrowserRequest {
  includeChunks?: boolean
  messages: MessageData[]
  config: WebLLMModelConfig
  options: Pick<StreamOptions, 'toolSpecs' | 'toolChoice' | 'dynamicTrailingBlocks'> & {
    systemPrompt?: string
  }
}
export interface BrowserCountRequest extends Omit<BrowserRequest, 'options'> {
  options: Omit<CountTokensOptions, 'systemPrompt'> & { systemPrompt?: string }
}
export type BrowserResult = (
  IteratorResult<ModelStreamEvent, void> | { error: { name: string; message: string } }
) & { chunks?: WebLLMChunk[] }

/** A private page controlled by the caller, such as a Playwright Page. */
export interface BrowserPage {
  evaluate<T, A>(run: (arg: A) => T | Promise<T>, arg: A): Promise<T>
}

/** Expose this object only on a trusted page controlled by your application. */
export interface BrowserBridge {
  count(request: BrowserCountRequest): Promise<number>
  start(request: BrowserRequest): Promise<string>
  next(id: string): Promise<BrowserResult>
  stop(id: string): Promise<void>
  close(): Promise<void>
}

declare global {
  interface Window {
    webllmBridge: BrowserBridge
  }
}
