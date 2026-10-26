import { afterEach, describe, expect, it, vi } from 'vitest'
import { BrowserWebLLMEmbeddings, WebLLMEmbeddings } from '../src/index.js'
import { browser } from '../fixtures/browser.js'

afterEach(() => vi.unstubAllGlobals())

describe.each(['local', 'browser'])('%s embedding input', (transport) => {
  it.each(['Orchid', '', new Set(['Orchid']), { 0: 'Orchid', length: 1 }, null, undefined])(
    'rejects a non-array document list before inference: %s',
    async (documents) => {
      const { engine, page, bridge } = browser()
      const create = vi.spyOn(engine.embeddings, 'create')
      const evaluate = vi.spyOn(page, 'evaluate')
      const model =
        transport === 'local'
          ? new WebLLMEmbeddings({ engine, modelId: 'embedding-model' })
          : new BrowserWebLLMEmbeddings(page, { modelId: 'embedding-model' })
      try {
        await expect(Reflect.apply(model.embedDocuments, model, [documents])).rejects.toThrow(
          'array of strings',
        )
        expect(create).not.toHaveBeenCalled()
        expect(evaluate).not.toHaveBeenCalled()
      } finally {
        await bridge.close()
      }
    },
  )
})
