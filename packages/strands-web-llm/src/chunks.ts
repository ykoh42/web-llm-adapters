import type { StopReason } from '@strands-agents/sdk'
import type { CompletionUsage } from '@mlc-ai/web-llm'

/** Only the fields needed to write Strands events. Native chunks stay available via onChunk. */
export interface StreamChunk {
  text: string
  reasoning: string
  toolText: string
  calls: ToolDelta[]
  stop: StopReason | undefined
  usage: CompletionUsage | undefined
}

export interface ToolDelta {
  index: number
  id: string | undefined
  name: string | undefined
  input: string | undefined
}

export function stopReason(reason: string | null | undefined): StopReason | undefined {
  switch (reason) {
    case 'stop':
      return 'endTurn'
    case 'length':
      return 'maxTokens'
    case 'abort':
      return 'cancelled'
    case 'tool_calls':
      return 'toolUse'
  }
}
