import { Embeddings } from '@langchain/core/embeddings'
import type { WebLLMEmbeddingsOptions } from '../embeddings/model.js'
import {
  defaultEmbeddingBatchSize,
  validateEmbeddingBatchSize,
  validateEmbeddingTexts,
} from '../embeddings/input.js'
import type { BrowserPage } from './types.js'

export type BrowserWebLLMEmbeddingsOptions = Omit<WebLLMEmbeddingsOptions, 'engine'>

/** LangChain and LangGraph embeddings from a caller-controlled browser. */
export class BrowserWebLLMEmbeddings extends Embeddings {
  private readonly config: Pick<WebLLMEmbeddingsOptions, 'modelId' | 'queryPrefix' | 'batchSize'>
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
    const input = [...texts]
    if (!input.length) return []
    validateEmbeddingTexts(input)
    const request = { config: this.config, texts: input, type }
    return this.caller.call(() =>
      this.page.evaluate((request) => {
        const bridge = window.webllmLangChainBridge
        if (!bridge)
          throw new Error(
            'Expose createBrowserBridge(engine) as window.webllmLangChainBridge before using BrowserWebLLMEmbeddings.',
          )
        return bridge.embed(request)
      }, request),
    )
  }
}
