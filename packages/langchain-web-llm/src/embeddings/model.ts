import { Embeddings, type EmbeddingsParams } from '@langchain/core/embeddings'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { modelError } from '../error.js'
import { readVectors, appendVectors } from './vectors.js'
import { withEngine } from '../../../../shared/web-llm/engine/access.js'
import {
  defaultEmbeddingBatchSize,
  validateEmbeddingBatchSize,
  copyEmbeddingTexts,
  validateEmbeddingText,
} from './input.js'

export interface WebLLMEmbeddingsOptions extends EmbeddingsParams {
  engine: MLCEngineInterface
  modelId: string
  /** Maximum documents per request; WebLLM splits to the artifact's GPU batch limit. Defaults to 32. */
  batchSize?: number
  /** Defaults to Snowflake Arctic's recommended search prefix for Snowflake models. */
  queryPrefix?: string
}

/** LangChain vectors from a loaded WebLLM embedding model. */
export class WebLLMEmbeddings extends Embeddings {
  readonly engine: MLCEngineInterface
  readonly modelId: string
  private readonly batchSize: number
  private readonly queryPrefix: string

  constructor({
    engine,
    modelId,
    queryPrefix,
    batchSize = defaultEmbeddingBatchSize,
    ...params
  }: WebLLMEmbeddingsOptions) {
    super({ ...params, maxRetries: params.maxRetries ?? 0 })
    if (!modelId) throw new Error('modelId must name a loaded WebLLM embedding model.')
    validateEmbeddingBatchSize(batchSize)
    this.engine = engine
    this.modelId = modelId
    this.batchSize = batchSize
    this.queryPrefix =
      queryPrefix ??
      (modelId.startsWith('snowflake-arctic-embed-')
        ? 'Represent this sentence for searching relevant passages: '
        : '')
  }

  override embedDocuments(documents: string[]): Promise<number[][]> {
    return this.embed(documents)
  }

  override async embedQuery(query: string): Promise<number[]> {
    validateEmbeddingText(query)
    const [vector] = await this.embed([this.queryPrefix + query])
    return vector!
  }

  private async embed(input: string[]): Promise<number[][]> {
    const texts = copyEmbeddingTexts(input)
    if (!texts.length) return []
    const vectors: number[][] = []
    for (let offset = 0; offset < texts.length; offset += this.batchSize) {
      const batch = texts.slice(offset, offset + this.batchSize)
      const response = await this.caller.call(async () => {
        try {
          return await withEngine(this.engine, async (engine) => {
            const response = await engine.embeddings.create({
              model: this.modelId,
              input: batch,
              encoding_format: 'float',
            })
            // Copy vectors before another engine request can reuse them.
            return {
              ...response,
              data: response.data.map((item) => ({ ...item, embedding: [...item.embedding] })),
            }
          })
        } catch (error) {
          throw modelError(error)
        }
      })
      appendVectors(vectors, readVectors(response, batch.length))
    }
    return vectors
  }
}
