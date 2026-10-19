import { Embeddings, type EmbeddingsParams } from '@langchain/core/embeddings'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { withEngine } from '../../../../shared/web-llm/engine/access.js'
import {
  defaultEmbeddingBatchSize,
  validateEmbeddingBatchSize,
  validateEmbeddingTexts,
} from './input.js'

export interface WebLLMEmbeddingsOptions extends EmbeddingsParams {
  engine: MLCEngineInterface
  modelId: string
  /** Maximum documents sent per WebLLM request. Defaults to 32. */
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
    validateEmbeddingTexts([query])
    const [vector] = await this.embed([this.queryPrefix + query])
    return vector!
  }

  private async embed(input: string[]): Promise<number[][]> {
    const texts = [...input]
    if (!texts.length) return []
    validateEmbeddingTexts(texts)
    return this.caller.call(() =>
      withEngine(this.engine, async (engine) => {
        const vectors: number[][] = []
        let dimensions: number | undefined

        for (let offset = 0; offset < texts.length; offset += this.batchSize) {
          const batch = texts.slice(offset, offset + this.batchSize)
          const response = await engine.embeddings.create({
            model: this.modelId,
            input: batch,
            encoding_format: 'float',
          })
          if (response.data.length !== batch.length)
            throw new Error('WebLLM returned the wrong number of embeddings.')

          const batchVectors: (number[] | undefined)[] = new Array(batch.length)
          for (const item of response.data) {
            if (
              !Number.isInteger(item.index) ||
              item.index < 0 ||
              item.index >= batch.length ||
              batchVectors[item.index] !== undefined
            )
              throw new Error('WebLLM returned invalid embedding indices.')
            if (!item.embedding.length || !item.embedding.every(Number.isFinite))
              throw new Error('WebLLM embeddings must contain finite numbers.')
            batchVectors[item.index] = item.embedding
          }

          for (const vector of batchVectors) {
            if (!vector) throw new Error('WebLLM returned invalid embedding indices.')
            if (dimensions !== undefined && vector.length !== dimensions)
              throw new Error('WebLLM embeddings must have matching dimensions.')
            dimensions = vector.length
            vectors.push(vector)
          }
        }
        return vectors
      }),
    )
  }
}
