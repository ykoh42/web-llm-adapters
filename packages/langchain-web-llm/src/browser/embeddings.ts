import { Embeddings } from '@langchain/core/embeddings'
import type { WebLLMEmbeddingsOptions } from '../embeddings/model.js'
import type { BrowserPage } from './types.js'

export type BrowserWebLLMEmbeddingsOptions = Omit<WebLLMEmbeddingsOptions, 'engine'>

/** LangChain and LangGraph embeddings from a caller-controlled browser. */
export class BrowserWebLLMEmbeddings extends Embeddings {
  private readonly config: Pick<WebLLMEmbeddingsOptions, 'modelId' | 'queryPrefix'>
  constructor(
    private readonly page: BrowserPage,
    { modelId, queryPrefix, ...params }: BrowserWebLLMEmbeddingsOptions,
  ) {
    super({ maxRetries: 0, ...params })
    if (!modelId) throw new Error('modelId must name a loaded WebLLM embedding model.')
    this.config = { modelId, ...(queryPrefix !== undefined && { queryPrefix }) }
  }
  override embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embed(texts, 'document')
  }
  override async embedQuery(text: string): Promise<number[]> {
    const [vector] = await this.embed([text], 'query')
    return vector!
  }
  private embed(texts: string[], type: 'query' | 'document') {
    return this.caller.call(() =>
      this.page.evaluate(
        (json) => window.webllmLangChainBridge.embed(JSON.parse(json)),
        JSON.stringify({ config: this.config, texts, type }),
      ),
    )
  }
}
