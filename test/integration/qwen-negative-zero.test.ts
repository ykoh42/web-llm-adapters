import { expect, it } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

const schema = {
  type: 'object' as const,
  properties: { value: { type: 'number' as const } },
  required: ['value'],
}
const tool = { name: 'lookup', description: 'Look up a value.', inputSchema: schema }

it.each([
  {
    modelId: 'Qwen3-4B-q4f16_1-MLC',
    content: '<tool_call>{"name":"lookup","arguments":{"value":-0.0}}</tool_call>',
  },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    content: '<tool_call><function=lookup><parameter=value>-0.0</parameter></function></tool_call>',
  },
])('$modelId delivers negative zero to every framework', async ({ modelId, content }) => {
  const { engine } = setupEngine(chunk({ content }, 'stop'))
  const strands = new WebLLMModel({ engine, modelId })
  const events = await collect(strands.stream([], { toolSpecs: [tool] }))
  const input = events
    .filter((event) => event.type === 'modelContentBlockDeltaEvent')
    .map((event) => (event.delta.type === 'toolUseInputDelta' ? event.delta.input : ''))
    .join('')
  expect(Object.is(JSON.parse(input).value, -0)).toBe(true)

  const langchain = new ChatWebLLM({ engine, modelId })
  const response = await langchain
    .bindTools([{ type: 'function', function: { name: tool.name, parameters: schema } }])
    .invoke('Look up the value.')
  expect(Object.is(response.tool_calls?.[0]?.args.value, -0)).toBe(true)

  const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
  const models = createModels()
  models.setProvider(provider)
  const responsePi = await models.complete(provider.getModels()[0]!, {
    messages: [{ role: 'user', content: 'Look up the value.', timestamp: 0 }],
    tools: [
      {
        name: tool.name,
        description: tool.description,
        parameters: Type.Object({ value: Type.Number() }),
      },
    ],
  })
  const call = responsePi.content.find((part) => part.type === 'toolCall')
  expect(call?.type).toBe('toolCall')
  if (call?.type !== 'toolCall') throw new Error('Missing tool call.')
  expect(Object.is(call.arguments.value, -0)).toBe(true)
})
