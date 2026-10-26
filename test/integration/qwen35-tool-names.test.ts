import { expect, it } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { Message, TextBlock } from '@strands-agents/sdk'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { collect, setupEngine } from '../../fixtures/web-llm.js'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const invalidNames = ['lookup>report', '<lookup']
const message = new Message({ role: 'user', content: [new TextBlock('Run a tool.')] })
const reason = 'Qwen3.5 tool names must be non-empty and cannot contain < or >.'

it.each(invalidNames)(
  'Strands rejects unrepresentable Qwen3.5 tool name %j before inference',
  async (name) => {
    const { engine, create } = setupEngine()
    const model = new WebLLMModel({ engine, modelId })

    await expect(
      collect(
        model.stream([message], {
          toolSpecs: [{ name, description: 'Read a value.' }],
        }),
      ),
    ).rejects.toThrow(reason)
    expect(create).not.toHaveBeenCalled()
  },
)

it.each(invalidNames)(
  'LangChain rejects unrepresentable Qwen3.5 tool name %j before inference',
  async (name) => {
    const { engine, create } = setupEngine()
    const model = new ChatWebLLM({ engine, modelId })

    await expect(
      model
        .bindTools([{ type: 'function', function: { name, description: 'Read a value.' } }])
        .invoke('Run a tool.'),
    ).rejects.toThrow(reason)
    expect(create).not.toHaveBeenCalled()
  },
)

it.each(invalidNames)(
  'Pi rejects unrepresentable Qwen3.5 tool name %j before inference',
  async (name) => {
    const { engine, create } = setupEngine()
    const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
    const models = createModels()
    models.setProvider(provider)
    const model = provider.getModels()[0]!
    const result = await models.complete(model, {
      messages: [{ role: 'user', content: 'Run a tool.', timestamp: 0 }],
      tools: [
        {
          name,
          description: 'Read a value.',
          parameters: Type.Object({}),
        },
      ],
    })

    expect(result.stopReason).toBe('error')
    expect(result.errorMessage).toContain(reason)
    expect(create).not.toHaveBeenCalled()
  },
)
