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
  const history = qwenMessages(messages, options.systemPrompt, tools, options.toolChoice)
  return { ...request, messages: history }
}

function qwenMessages(
  messages: Message[],
  systemPrompt?: StreamOptions['systemPrompt'],
  tools: ToolSpec[] = [],
  choice?: ToolChoice,
) {
  const output = formatMessages([], undefined)
  const system = formatSystemPrompt(systemPrompt)
  const prompt = [system, toolPrompt(tools, choice)].filter(Boolean).join('\n\n')
  if (prompt) output.push({ role: 'system', content: prompt })

  for (const message of messages) {
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
      const parts = [
        ...thoughts.map((block) => `<think>${block.text ?? ''}</think>`),
        typeof text === 'string' ? text : '',
        ...content
          .filter((block) => block.type === 'toolUseBlock')
          .map(
            (block) =>
              `<tool_call>\n${JSON.stringify({ name: block.name, arguments: block.input })}\n</tool_call>`,
          ),
      ].filter(Boolean)
      output.push({ role: 'assistant', content: parts.join('\n') || null })
      continue
    }

    for (const turn of formatUser(message.content)) {
      if (turn.role === 'tool') {
        output.push({ role: 'user', content: `<tool_response>\n${turn.content}\n</tool_response>` })
      } else {
        output.push(turn)
      }
    }
  }
  return output
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
