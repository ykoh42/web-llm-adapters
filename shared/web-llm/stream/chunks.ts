import { ModelError, type StopReason } from '../types.js'
import type { CompletionUsage } from '@mlc-ai/web-llm'

export interface ChunkOptions<T> {
  onChunk?: ((chunk: T) => void) | undefined
  signal?: AbortSignal | undefined
}

/** Common output for framework adapters. Native chunks stay available via onChunk. */
export interface StreamChunk {
  text: string
  reasoning: string
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

export function singleChoice<T extends { index: number }>(choices: readonly T[]): T | undefined {
  const choice = choices[0]
  if (choices.length > 1 || (choice && choice.index !== 0))
    throw new ModelError('WebLLM adapters need one response per request.')
  return choice
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
