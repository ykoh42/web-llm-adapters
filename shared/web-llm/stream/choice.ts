import { ModelError, type ToolChoice, type StopReason } from '../types.js'

/** Requiring the only registered tool is equivalent to selecting it by name. */
export function singleToolChoice(
  choice: ToolChoice,
  tools: readonly { name: string }[],
): ToolChoice {
  return 'any' in choice && tools.length === 1 ? { tool: { name: tools[0]!.name } } : choice
}

/** A model's choice must not bypass the caller's tool restriction. */
export function checkToolChoice(
  names: Iterable<string | undefined>,
  choice: ToolChoice | 'none' | undefined,
  stop: StopReason,
  toolNames?: readonly string[],
) {
  if (!choice || stop === 'maxTokens' || stop === 'cancelled') return
  if (choice !== 'none' && 'auto' in choice) return
  const calls = [...names]
  if (choice === 'none') {
    if (calls.length) throw new ModelError('WebLLM returned a tool despite tool_choice none.')
    return
  }
  if (!calls.length) throw new ModelError('WebLLM did not return the required tool call.')
  if ('any' in choice && toolNames?.length === 1) choice = { tool: { name: toolNames[0]! } }
  if ('tool' in choice && calls.some((name) => name !== choice.tool.name))
    throw new ModelError('WebLLM returned a tool outside tool_choice.')
}
