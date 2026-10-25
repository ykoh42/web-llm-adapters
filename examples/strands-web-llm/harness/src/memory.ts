import { sampling } from '../../sampling.ts'
import type { Page } from '@playwright/test'
import { LocalFileStorage } from '@strands-agents/sdk/storage'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { BrowserModel, WebLLMEmbeddingSearchStrategy } from 'strands-web-llm'

export const embeddingModelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
const queryPrefix = 'Represent this sentence for searching relevant passages: '

/** Keep extraction JSON valid without constraining ordinary summaries. */
export function createMemory(
  page: Page,
  modelId: string,
  contextWindow: number,
  dir = '.agent/memory',
) {
  return new FileMemoryStore({
    name: 'memory',
    storage: new LocalFileStorage(dir).namespace(''),
    extraction: { model: createMemoryModel(page, modelId, contextWindow) },
  })
}

/** Store durable facts with the same Strands store API and rank them by WebLLM vectors. */
export function createEmbeddingMemory(
  page: Page,
  chatModelId: string,
  contextWindow: number,
  dir: string,
) {
  const strategy = new WebLLMEmbeddingSearchStrategy({
    includeData: false,
    embed: async (texts, inputType) =>
      page.evaluate(
        ({ embeddingModelId, texts, inputType, queryPrefix }) =>
          window.webllmBridge.withEngine(async (engine) => {
            const input = texts.map((text) =>
              inputType === 'query' ? `${queryPrefix}${text}` : text,
            )
            const result = await engine.embeddings.create({ model: embeddingModelId, input })
            return [...result.data]
              .sort((left, right) => left.index - right.index)
              .map(({ embedding }) => embedding)
          }),
        { embeddingModelId, texts: [...texts], inputType, queryPrefix },
      ),
  })

  return new FileMemoryStore({
    name: 'semantic-memory',
    description: 'Durable facts ranked with WebLLM embeddings.',
    storage: new LocalFileStorage(dir, { searchStrategy: strategy }).namespace(''),
    extraction: { model: createMemoryModel(page, chatModelId, contextWindow) },
  })
}

function createMemoryModel(page: Page, modelId: string, contextWindow: number) {
  const settings = sampling(modelId, false)
  return new BrowserModel(page, {
    modelId,
    thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
    contextWindowLimit: contextWindow,
    maxTokens: 4096,
    ...settings,
    request: {
      ...settings.request,
      response_format: {
        type: 'json_object',
        schema: JSON.stringify({
          type: 'array',
          items: {
            type: 'object',
            properties: { content: { type: 'string' } },
            required: ['content'],
            additionalProperties: false,
          },
        }),
      },
    },
  })
}
