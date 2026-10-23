import { expect, it, vi } from 'vitest'
import { BrowserModel, WebLLMModel, type BrowserPage } from '../src/index.js'
import { setup } from '../fixtures/engine.js'

it.each([false, true])('requires a model ID before inference (browser: %s)', (browser) => {
  const { engine, create } = setup()
  const page: BrowserPage = {
    async evaluate() {
      throw new Error('Unexpected browser call.')
    },
  }
  const evaluate = vi.spyOn(page, 'evaluate')
  const makeModel = (modelId: string) =>
    browser ? new BrowserModel(page, { modelId }) : new WebLLMModel({ engine, modelId })

  expect(() => makeModel('')).toThrow('modelId must name a loaded WebLLM model.')

  const model = makeModel('test-model')
  const config = model.getConfig()
  expect(() => model.updateConfig({ modelId: '', maxTokens: 123 })).toThrow(
    'modelId must name a loaded WebLLM model.',
  )
  expect(model.getConfig()).toEqual(config)
  expect(model.modelId).toBe('test-model')

  model.updateConfig({ modelId: 'other-model' })
  expect(model.modelId).toBe('other-model')
  expect(create).not.toHaveBeenCalled()
  expect(evaluate).not.toHaveBeenCalled()
})
