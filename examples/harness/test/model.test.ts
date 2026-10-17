import { expect, it } from 'vitest'
import { createHarness } from '@strands-agents/harness'
import { BrowserModel, type BrowserPage } from 'strands-webllm'

it('accepts the public package model as a Harness model', async () => {
  const page: BrowserPage = {
    evaluate: async () => {
      throw new Error('No browser call expected')
    },
  }
  const model = new BrowserModel(page, {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    contextWindowLimit: 16384,
  })
  const agent = await createHarness({
    model,
    builtinTools: [],
    builtinPlugins: [],
    session: false,
    memory: false,
    skills: false,
    contextManager: false,
    caching: false,
    backgroundTasks: false,
    printer: false,
  })
  try {
    expect(agent.model).toBe(model)
  } finally {
    await agent.shutdown()
  }
})
