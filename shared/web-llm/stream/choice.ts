import { ModelError, type ToolChoice, type StopReason } from '../types.js'

/** Requiring the only registered tool is equivalent to selecting it by name. */
export function singleToolChoice(
  choice: ToolChoice,
  tools: readonly { name: string }[],
): ToolChoice {
  return 'any' in choice && tools.length === 1 ? { tool: { name: tools[0]!.name } } : choice
}

/** Tool result messages use call IDs to associate each result with its request. */
export function checkToolCallIds(calls: Iterable<{ readonly id?: string | undefined }>) {
  const ids = new Set<string>()
  for (const { id } of calls) {
    if (!id) continue
    if (ids.has(id)) throw new ModelError('WebLLM returned duplicate tool call IDs.')
    ids.add(id)
  }
}

/** A model's choice must not bypass the caller's tool restriction. */
export function checkToolChoice(
  calls: readonly { name: string }[],
  choice: ToolChoice | 'none' | undefined,
  stop: StopReason,
  toolNames?: readonly string[],
) {
  if (stop === 'maxTokens' || stop === 'cancelled') return
  if (choice === 'none') {
    if (calls.length) throw new ModelError('WebLLM returned a tool despite tool_choice none.')
    return
  }
  if (!choice || 'auto' in choice) return
  if (!calls.length) throw new ModelError('WebLLM did not return the required tool call.')
  if ('any' in choice && toolNames?.length === 1) choice = { tool: { name: toolNames[0]! } }
  if ('tool' in choice && calls.some((call) => call.name !== choice.tool.name))
    throw new ModelError('WebLLM returned a tool outside tool_choice.')
}
