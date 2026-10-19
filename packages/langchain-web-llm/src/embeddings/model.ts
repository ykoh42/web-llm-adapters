import { Embeddings, type EmbeddingsParams } from '@langchain/core/embeddings'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { withEngine } from '../../../../shared/web-llm/engine/access.js'

export interface WebLLMEmbeddingsOptions extends EmbeddingsParams {
  engine: MLCEngineInterface
  modelId: string
  /** Defaults to Snowflake Arctic's recommended search prefix for Snowflake models. */
  queryPrefix?: string
}

/** LangChain vectors from a loaded WebLLM embedding model. */
export class WebLLMEmbeddings extends Embeddings {
  readonly engine: MLCEngineInterface
  readonly modelId: string
  private readonly queryPrefix: string

  constructor({ engine, modelId, queryPrefix, ...params }: WebLLMEmbeddingsOptions) {
    super({ maxRetries: 0, ...params })
    if (!modelId) throw new Error('modelId must name a loaded WebLLM embedding model.')
    this.engine = engine
    this.modelId = modelId
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
    if (!query.trim()) throw new Error('WebLLM cannot embed empty text.')
    const [vector] = await this.embed([this.queryPrefix + query])
    return vector!
  }

  private async embed(input: string[]): Promise<number[][]> {
    if (!input.length) return []
    if (input.some((text) => !text.trim())) throw new Error('WebLLM cannot embed empty text.')
    return this.caller.call(() =>
      withEngine(this.engine, async (engine) => {
        const response = await engine.embeddings.create({
          model: this.modelId,
          input,
          encoding_format: 'float',
        })
        if (response.data.length !== input.length)
          throw new Error('WebLLM returned the wrong number of embeddings.')
        const vectors: number[][] = []
        for (const item of response.data) {
          if (
            !Number.isInteger(item.index) ||
            item.index < 0 ||
            item.index >= input.length ||
            vectors[item.index]
          )
            throw new Error('WebLLM returned invalid embedding indices.')
          if (!item.embedding.length || !item.embedding.every(Number.isFinite))
            throw new Error('WebLLM embeddings must contain finite numbers.')
          vectors[item.index] = item.embedding
        }
        if (vectors.some((vector) => vector.length !== vectors[0]!.length))
          throw new Error('WebLLM embeddings must have matching dimensions.')
        return vectors
      }),
    )
  }
}
