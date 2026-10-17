import type { Page } from '@playwright/test'
import { LocalFileStorage } from '@strands-agents/sdk/storage'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { BrowserModel } from 'strands-webllm'

/** Keep extraction JSON valid without constraining ordinary summaries. */
export function createMemory(
  page: Page,
  modelId: string,
  contextWindow: number,
  dir = '.agent/memory',
) {
  const model = new BrowserModel(page, {
    modelId,
    contextWindowLimit: contextWindow,
    maxTokens: 512,
    temperature: 0,
    // Leave thinking unset so Gemma uses WebLLM's native JSON path.
    request: {
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
  return new FileMemoryStore({
    name: 'memory',
    storage: new LocalFileStorage(dir).namespace(''),
    extraction: { model },
  })
}
