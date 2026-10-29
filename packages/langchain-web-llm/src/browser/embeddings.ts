import { restoreBrowserError } from './errors.js'
import { readBrowserError } from '../../../../shared/browser/errors.js'
import { appendVectors } from '../embeddings/vectors.js'
import { Embeddings } from '@langchain/core/embeddings'
import type { WebLLMEmbeddingsOptions } from '../embeddings/model.js'
import {
  defaultEmbeddingBatchSize,
  validateEmbeddingBatchSize,
  copyEmbeddingTexts,
} from '../embeddings/input.js'
import type { BrowserPage } from './types.js'

export type BrowserWebLLMEmbeddingsOptions = Omit<WebLLMEmbeddingsOptions, 'engine'>

/** LangChain and LangGraph embeddings from a caller-controlled browser. */
export class BrowserWebLLMEmbeddings extends Embeddings {
  private readonly config: Pick<WebLLMEmbeddingsOptions, 'modelId' | 'queryPrefix'> & {
    batchSize: number
  }
  constructor(
    private readonly page: BrowserPage,
    {
      modelId,
      queryPrefix,
      batchSize = defaultEmbeddingBatchSize,
      ...params
    }: BrowserWebLLMEmbeddingsOptions,
  ) {
    super({ ...params, maxRetries: params.maxRetries ?? 0 })
    if (!modelId) throw new Error('modelId must name a loaded WebLLM embedding model.')
    validateEmbeddingBatchSize(batchSize)
    this.config = {
      modelId,
      batchSize,
      ...(queryPrefix !== undefined && { queryPrefix }),
    }
  }
  override embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embed(texts, 'document')
  }
  override async embedQuery(text: string): Promise<number[]> {
    const [vector] = await this.embed([text], 'query')
    return vector!
  }
  private async embed(texts: string[], type: 'query' | 'document'): Promise<number[][]> {
    const input = copyEmbeddingTexts(texts)
    if (!input.length) return []
    const vectors: number[][] = []
    for (let offset = 0; offset < input.length; offset += this.config.batchSize) {
      const request = {
        hostRetries: true,
        config: this.config,
        texts: input.slice(offset, offset + this.config.batchSize),
        type,
      }
      const batch = await this.caller.call(async () => {
        const result = await this.page.evaluate((request) => {
          const bridge = window.webllmLangChainBridge
          if (!bridge)
            throw new Error(
              'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserWebLLMEmbeddings.',
            )
          return bridge.embed(request)
        }, request)
        if (!Array.isArray(result)) throw restoreBrowserError(readBrowserError(result))
        return result
      })
      appendVectors(vectors, batch)
    }
    return vectors
  }
}
