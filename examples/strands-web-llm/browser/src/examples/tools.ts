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
    contextWindowLimit: thinking ? 4096 : 2048,
    maxTokens: thinking ? 1024 : 256,
    temperature: 0,
    ...(thinking && { request: { extra_body: { enable_thinking: true } } }),
  })
  const agent = new Agent({ model, tools: [clock], printer: false })
  // These models repeat tool calls instead of answering after a tool result.
  const { functionCallingModelIds } = await import('@mlc-ai/web-llm')
  if (functionCallingModelIds.includes(input.modelId))
    agent.addHook(AfterToolsEvent, (event) => {
      event.endTurn = [new TextBlock('')]
    })
  await runAgent(agent, input.prompt, input)
}
