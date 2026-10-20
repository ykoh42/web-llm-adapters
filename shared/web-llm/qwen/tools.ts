import { ModelError, type ToolSpec, type ToolChoice } from '../types.js'
import { stringifyJSON } from '../json.js'
import { nativeTools } from '../native.js'

// Keep Hermes JSON compact: spacing changes Qwen3 tool-loop behavior in WebLLM.
export function formatCall(name: string, input: unknown) {
  return `<tool_call>\n{"name": ${JSON.stringify(name)}, "arguments": ${stringifyJSON(input)}}\n</tool_call>`
}

export function toolPrompt(tools: ToolSpec[], choice?: ToolChoice) {
  if (choice && !('auto' in choice) && !tools.length)
    throw new ModelError('A required Qwen tool choice needs at least one registered tool.')
  if (!tools.length) return ''
  if (choice && 'tool' in choice && !tools.some((tool) => tool.name === choice.tool.name)) {
    throw new ModelError('The chosen Qwen tool is not registered.')
  }
  const rules = [
    '# Tools',
    '',
    'You may call one or more functions to assist with the user query.',
    '',
    'You are provided with function signatures within <tools></tools> XML tags:',
    '<tools>',
    ...nativeTools(tools).map((tool) => stringifyJSON(tool)),
    '</tools>',
    '',
    'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:',
    '<tool_call>',
    '{"name": <function-name>, "arguments": <args-json-object>}',
    '</tool_call>',
  ]
  if (choice && 'any' in choice) rules.push('You must call at least one function.')
  if (choice && 'tool' in choice) rules.push(`Call only the ${choice.tool.name} function.`)
  return rules.join('\n')
}
