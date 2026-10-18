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
import { appendQwenTurns, lastQwenQuery } from './history.js'

const qwen25SystemPrompt = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

const modelIds = new Set([
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen3-0.6B-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
])

export function usesQwenProtocol(config: WebLLMModelConfig) {
  return modelIds.has(config.modelId)
}

export function qwenRequest(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
): ChatCompletionRequestStreaming {
  const tools = options.toolSpecs ?? []
  const hasToolHistory = messages.some((message) =>
    message.content.some(
      (block) => block.type === 'toolUseBlock' || block.type === 'toolResultBlock',
    ),
  )
  if (tools.length && hasStructuredResponse(config))
    throw new ModelError('Qwen tool calls cannot use a structured response format.')
  if (
    config.modelId === 'Qwen3-1.7B-q4f16_1-MLC' &&
    (tools.length > 0 || options.toolChoice !== undefined || hasToolHistory)
  )
    throw new ModelError('Tool calling is not supported for Qwen3-1.7B-q4f16_1-MLC.')

  const request = generationOptions(config)
  delete request.tool_choice
  const history = qwenMessages(
    config.modelId.startsWith('Qwen3-'),
    messages,
    options.systemPrompt,
    tools,
    options.toolChoice,
  )
  return { ...request, messages: history }
}

function qwenMessages(
  isQwen3: boolean,
  messages: Message[],
  systemPrompt?: StreamOptions['systemPrompt'],
  tools: ToolSpec[] = [],
  choice?: ToolChoice,
) {
  const output = formatMessages([], undefined)
  const system = formatSystemPrompt(systemPrompt)
  const systemParts = []
  if (system !== undefined) systemParts.push(system)
  else if (!isQwen3) systemParts.push(qwen25SystemPrompt)
  if (tools.length) systemParts.push(toolPrompt(tools, choice))
  if (systemParts.length) output.push({ role: 'system', content: systemParts.join('\n\n') })

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
        throw new ModelError('Qwen cannot use signed or redacted reasoning.')
      const content = message.content.filter((block) => block.type !== 'reasoningBlock')
      const text = formatAssistant(content.filter((block) => block.type !== 'toolUseBlock')).content
      const thought = isQwen3
        ? thoughts
            .map((block) => block.text ?? '')
            .join('\n')
            .trim()
        : ''
      const reasoning =
        isQwen3 && index > lastQuery && thought ? `<think>\n${thought}\n</think>\n\n` : ''
      const answer = typeof text === 'string' ? text : ''
      const calls = content.filter((block) => block.type === 'toolUseBlock')
      const callText = calls
        .map(
          (call, callIndex) =>
            `${callIndex === 0 && answer ? '\n' : callIndex ? '\n' : ''}${formatCall(call.name, call.input)}`,
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

// Keep Hermes JSON compact: spacing changes Qwen3 tool-loop behavior in WebLLM.
function formatCall(name: string, input: unknown) {
  return `<tool_call>\n{"name": ${JSON.stringify(name)}, "arguments": ${JSON.stringify(input)}}\n</tool_call>`
}

function toolPrompt(tools: ToolSpec[], choice?: ToolChoice) {
  if (!tools.length) return ''
  if (choice && 'any' in choice) {
    // Qwen can choose any declared tool; its prompt format has no native choice field.
  } else if (choice && 'tool' in choice && !tools.some((tool) => tool.name === choice.tool.name)) {
    throw new ModelError('The chosen Qwen tool is not registered.')
  }
  const rules = [
    '# Tools',
    '',
    'You may call one or more functions to assist with the user query.',
    '',
    'You are provided with function signatures within <tools></tools> XML tags:',
    '<tools>',
    ...formatTools(tools).map((tool) => JSON.stringify(tool)),
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
