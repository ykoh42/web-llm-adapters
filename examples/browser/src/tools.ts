import { AfterToolsEvent, Agent, FunctionTool } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-webllm'
import { runAgent, type ExampleInput } from './output.js'

export async function tools(input: ExampleInput) {
  const clock = new FunctionTool({
    name: 'clock',
    description: 'Get the current UTC date and time.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    callback: () => new Date().toISOString(),
  })
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    contextWindowLimit: 2048,
    maxTokens: 256,
    temperature: 0,
  })
  const agent = new Agent({ model, tools: [clock], printer: false })
  // Hermes tool mode always uses a tool JSON grammar. End after executing the tools.
  agent.addHook(AfterToolsEvent, (event) => {
    event.endTurn = 'Tool call finished. See the result in Details.'
  })
  await runAgent(agent, input.prompt, input)
}
