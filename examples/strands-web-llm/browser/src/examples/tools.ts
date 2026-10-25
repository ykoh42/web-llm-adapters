import { sampling } from '../../../sampling.js'
import { AfterToolsEvent, Agent, FunctionTool, TextBlock } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-web-llm'
import { runAgent, type ExampleInput } from './output.js'

export async function tools(input: ExampleInput, thinking = false) {
  const clock = new FunctionTool({
    name: 'clock',
    description: 'Get the current UTC date and time.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    callback: () => new Date().toISOString(),
  })
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    thinkingPrefilled: input.modelId.startsWith('Qwen3.5-'),
    contextWindowLimit: thinking ? 8192 : 2048,
    maxTokens: thinking ? 4096 : 256,
    ...sampling(input.modelId, thinking),
  })
  const agent = new Agent({ model, tools: [clock], printer: false })
  // WebLLM requires tool-call JSON on every native Hermes request.
  const { functionCallingModelIds } = await import('@mlc-ai/web-llm')
  if (functionCallingModelIds.includes(input.modelId))
    agent.addHook(AfterToolsEvent, (event) => {
      event.endTurn = [new TextBlock('')]
    })
  await runAgent(agent, input.prompt, input)
}
