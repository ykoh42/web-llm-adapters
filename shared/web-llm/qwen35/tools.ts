import { ModelError, type ToolSpec, type ToolChoice } from '../types.js'
import { stringifyJSON } from '../json.js'
import { nativeTools } from '../native.js'

export function toolPrompt(tools: ToolSpec[], choice?: ToolChoice) {
  if (choice && !('auto' in choice) && !tools.length)
    throw new ModelError('A required Qwen3.5 tool choice needs at least one registered tool.')
  if (!tools.length) return ''
  if (choice && 'tool' in choice && !tools.some((tool) => tool.name === choice.tool.name))
    throw new ModelError('The chosen Qwen3.5 tool is not registered.')
  const rules = [
    '# Tools',
    '',
    'You have access to the following functions:',
    '',
    '<tools>',
    ...nativeTools(tools).map(templateJson),
    '</tools>',
    '',
    'If you choose to call a function ONLY reply in the following format with NO suffix:',
    '',
    '<tool_call>',
    '<function=example_function_name>',
    '<parameter=example_parameter_1>',
    'value_1',
    '</parameter>',
    '<parameter=example_parameter_2>',
    'This is the value for the second parameter',
    'that can span',
    'multiple lines',
    '</parameter>',
    '</function>',
    '</tool_call>',
    '',
    '<IMPORTANT>',
    'Reminder:',
    '- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags',
    '- Required parameters MUST be specified',
    '- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after',
    '- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls',
    '</IMPORTANT>',
  ]
  if (choice && 'any' in choice) rules.push('You must call at least one function.')
  if (choice && 'tool' in choice) rules.push(`Call only the ${choice.tool.name} function.`)
  return rules.join('\n')
}

/** Match the template's JSON separators without changing strings. */
function templateJson(value: unknown) {
  return stringifyJSON(value).replace(/"(?:\\.|[^"\\])*"|[:,]/g, (token) =>
    token === ',' || token === ':' ? `${token} ` : token,
  )
}

export function formatCall(name: string, input: unknown) {
  // Match the JSON values sent by native requests and browser transport.
  const args: unknown = JSON.parse(stringifyJSON(input))
  if (!isRecord(args)) throw new ModelError('Qwen3.5 tool arguments must be an object.')
  const parameters = Object.entries(args).map(([key, value]) => {
    const text = typeof value === 'string' ? value : templateJson(value)
    return `<parameter=${key}>\n${text}\n</parameter>`
  })
  return `<tool_call>\n<function=${name}>\n${parameters.join('\n')}${parameters.length ? '\n' : ''}</function>\n</tool_call>`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
