import { ModelError, type StopReason } from '../types.js'
import type { ChunkCallback } from '../callback.js'
import type { ChatCompletionChunk, Completion, CompletionUsage } from '@mlc-ai/web-llm'

export interface ChunkOptions<T> {
  onChunk?: ChunkCallback<T> | undefined
  /** Defaults to true; set false only for internal observers that do not mutate chunks. */
  onChunkMayMutate?: boolean | undefined
  signal?: AbortSignal | undefined
}

/** Common output for framework adapters. Native chunks stay available via onChunk. */
export interface StreamChunk {
  text: string
  reasoning: string
  calls: readonly ToolDelta[]
  stop: StopReason | undefined
  usage: CompletionUsage | undefined
}

export interface ToolDelta {
  index: number
  id: string | undefined
  name: string | undefined
  input: string | undefined
}

export const noToolCalls: readonly ToolDelta[] = Object.freeze([])

/** Copy parser inputs before a native-chunk observer can mutate them. */
export function snapshotChatChunk(chunk: ChatCompletionChunk) {
  return {
    choices: chunk.choices.map((choice) => ({
      ...choice,
      delta: {
        ...choice.delta,
        ...(choice.delta.tool_calls && {
          tool_calls: choice.delta.tool_calls.map((call) => ({
            ...call,
            ...(call.function && { function: { ...call.function } }),
          })),
        }),
      },
    })),
    ...(chunk.usage && { usage: structuredClone(chunk.usage) }),
  } satisfies Pick<ChatCompletionChunk, 'choices' | 'usage'>
}

/** Copy parser inputs before a native-chunk observer can mutate them. */
export function snapshotCompletion(chunk: Completion) {
  return {
    choices: chunk.choices.map((choice) => ({ ...choice })),
    ...(chunk.usage && { usage: structuredClone(chunk.usage) }),
  } satisfies Pick<Completion, 'choices' | 'usage'>
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
  return undefined
}
