import {
  ModelError,
  type Message,
  type StreamOptions,
  type ToolChoice,
  type ToolSpec,
} from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig } from '../types.js'
import {
  formatAssistant,
  formatMessages,
  formatSystemPrompt,
  formatTools,
  formatUser,
} from '../chat/messages.js'
import { generationOptions, hasStructuredResponse } from '../request.js'
import { appendQwenTurns, lastQwenQuery } from '../qwen/history.js'

const modelIds = new Set([
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-9B-q4f16_1-MLC',
])

export function usesQwen35Protocol(config: WebLLMModelConfig) {
  return modelIds.has(config.modelId)
}

export function qwen35Request(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
): ChatCompletionRequestStreaming {
  if ((options.toolSpecs?.length ?? 0) && hasStructuredResponse(config))
    throw new ModelError('Qwen3.5 tool calls cannot use a structured response format.')
  const request = generationOptions(config)
  delete request.tool_choice
  return {
    ...request,
    messages: qwen35Messages(
      messages,
      options.systemPrompt,
      options.toolSpecs ?? [],
      options.toolChoice,
    ),
  }
}

function qwen35Messages(
  messages: Message[],
  systemPrompt?: StreamOptions['systemPrompt'],
  tools: ToolSpec[] = [],
  choice?: ToolChoice,
) {
  const output = formatMessages([], undefined)
  const system = formatSystemPrompt(systemPrompt)
  const prompt = tools.length
    ? [toolPrompt(tools, choice), ...(system ? [system] : [])].join('\n\n')
    : system
  if (prompt) output.push({ role: 'system', content: prompt })

  const lastQuery = lastQwenQuery(messages)
  let previousWasToolResult = false
  for (const [index, message] of messages.entries()) {
    if (message.role === 'assistant') {
      const thoughts = message.content.filter((block) => block.type === 'reasoningBlock')
      if (
        thoughts.some(
          (block) => block.signature !== undefined || block.redactedContent !== undefined,
        )
      )
        throw new ModelError('Qwen3.5 cannot use signed or redacted reasoning.')
      const content = message.content.filter((block) => block.type !== 'reasoningBlock')
      const text = formatAssistant(content.filter((block) => block.type !== 'toolUseBlock')).content
      const calls = content
        .filter((block) => block.type === 'toolUseBlock')
        .map((block) => formatCall(block.name, block.input))
      const thought = thoughts
        .map((block) => block.text ?? '')
        .join('\n')
        .trim()
      const reasoning = index > lastQuery && thought ? `<think>\n${thought}\n</think>\n\n` : ''
      const answer = typeof text === 'string' ? text : ''
      const callText = calls
        .map(
          (call, callIndex) =>
            `${callIndex === 0 && answer ? '\n\n' : callIndex ? '\n' : ''}${call}`,
        )
        .join('')
      output.push({ role: 'assistant', content: `${reasoning}${answer}${callText}` || null })
      previousWasToolResult = false
      continue
    }

    previousWasToolResult = appendQwenTurns(
      output,
      formatUser(message.content),
      previousWasToolResult,
    )
  }
  return output
}

function toolPrompt(tools: ToolSpec[], choice?: ToolChoice) {
  if (!tools.length) return ''
  if (choice && 'tool' in choice && !tools.some((tool) => tool.name === choice.tool.name))
    throw new ModelError('The chosen Qwen3.5 tool is not registered.')
  const rules = [
    '# Tools',
    '',
    'You have access to the following functions:',
    '',
    '<tools>',
    ...formatTools(tools).map(templateJson),
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
  return JSON.stringify(value).replace(/"(?:\\.|[^"\\])*"|[:,]/g, (token) =>
    token === ',' || token === ':' ? `${token} ` : token,
  )
}

function formatCall(name: string, input: unknown) {
  if (!isRecord(input)) throw new ModelError('Qwen3.5 tool arguments must be an object.')
  const parameters = Object.entries(input).map(([key, value]) => {
    const text = typeof value === 'string' ? value : templateJson(value)
    return `<parameter=${key}>\n${text}\n</parameter>`
  })
  return `<tool_call>\n<function=${name}>\n${parameters.join('\n')}${parameters.length ? '\n' : ''}</function>\n</tool_call>`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
