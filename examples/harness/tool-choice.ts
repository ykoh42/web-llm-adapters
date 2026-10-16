import { InvokeModelStage, type Agent } from '@strands-agents/sdk'

/** Choose the next tool. The model still writes its arguments and final answer. */
export function useTool(agent: Agent, name: string) {
  let first = true
  agent.addMiddleware(InvokeModelStage.Input, async (context) => {
    if (!first) return context
    first = false
    return { ...context, toolChoice: { tool: { name } } }
  })
}
