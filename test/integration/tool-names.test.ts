import { expect, it, vi } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { Message, TextBlock } from '@strands-agents/sdk'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { collect, setupEngine } from '../../fixtures/web-llm.js'

const modelIds = ['test-model', 'Qwen3-4B', 'Qwen3.5-4B', 'gemma-4-e2b-it']
const toolSpecs = [
  { name: 'lookup', description: 'Look up a value.' },
  { name: 'lookup', description: 'Look up another value.' },
]
const emptyPiTool = [{ name: '', description: 'Look up a value.', parameters: Type.Object({}) }]
const message = new Message({ role: 'user', content: [new TextBlock('Run a tool.')] })
const reason = 'WebLLM tool names must be unique and non-empty.'

it.each(modelIds)(
  'Strands rejects duplicate tool names for %s before inference',
  async (modelId) => {
    const { engine, create } = setupEngine()
    const completions = vi.spyOn(engine.completions, 'create')
    const model = new WebLLMModel({ engine, modelId })

    await expect(collect(model.stream([message], { toolSpecs }))).rejects.toThrow(reason)
    expect(create).not.toHaveBeenCalled()
    expect(completions).not.toHaveBeenCalled()
  },
)

it.each(modelIds)('Strands rejects empty tool names for %s before inference', async (modelId) => {
  const { engine, create } = setupEngine()
  const completions = vi.spyOn(engine.completions, 'create')
  const model = new WebLLMModel({ engine, modelId })

  await expect(
    collect(
      model.stream([message], { toolSpecs: [{ name: '', description: 'Look up a value.' }] }),
    ),
  ).rejects.toThrow(reason)
  expect(create).not.toHaveBeenCalled()
  expect(completions).not.toHaveBeenCalled()
})

it.each(modelIds)('Pi rejects empty tool names for %s before inference', async (modelId) => {
  const { engine, create } = setupEngine()
  const completions = vi.spyOn(engine.completions, 'create')
  const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
  const models = createModels()
  models.setProvider(provider)
  const model = provider.getModels()[0]!
  const result = await models.complete(model, {
    messages: [{ role: 'user', content: 'Run a tool.', timestamp: 0 }],
    tools: emptyPiTool,
  })

  expect(result.stopReason).toBe('error')
  expect(result.errorMessage).toContain(reason)
  expect(create).not.toHaveBeenCalled()
  expect(completions).not.toHaveBeenCalled()
})
