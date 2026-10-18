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
        throw new ModelError('Qwen3.5 cannot use signed or redacted reasoning.')
      const content = message.content.filter((block) => block.type !== 'reasoningBlock')
      const text = formatAssistant(content.filter((block) => block.type !== 'toolUseBlock')).content
      const calls = content
        .filter((block) => block.type === 'toolUseBlock')
        .map((block) => formatCall(block.name, block.input))
      const parts = [
        ...thoughts.map((block) => `<think>${block.text ?? ''}</think>`),
        typeof text === 'string' ? text : '',
        ...calls,
      ].filter(Boolean)
      output.push({ role: 'assistant', content: parts.join('\n') || null })
      continue
    }

    for (const turn of formatUser(message.content)) {
      if (turn.role === 'tool') {
        output.push({ role: 'user', content: `<tool_response>${turn.content}</tool_response>` })
      } else {
        output.push(turn)
      }
    }
  }
  return output
}

function toolPrompt(tools: ToolSpec[], choice?: ToolChoice) {
  if (!tools.length) return ''
  if (choice && 'tool' in choice && !tools.some((tool) => tool.name === choice.tool.name))
    throw new ModelError('The chosen Qwen3.5 tool is not registered.')
  const rules = [
    '# Tools',
    'You have access to the following functions:',
    '<tools>',
    ...formatTools(tools).map((tool) => JSON.stringify(tool)),
    '</tools>',
    'If you choose to call a function, only reply in this format with no suffix:',
    '<tool_call>',
    '<function=function_name>',
    '<parameter=parameter_name>value</parameter>',
    '</function>',
    '</tool_call>',
    'Wait for the tool result before answering.',
  ]
  if (choice && 'any' in choice) rules.push('You must call at least one function.')
  if (choice && 'tool' in choice) rules.push(`Call only the ${choice.tool.name} function.`)
  return rules.join('\n')
}

function formatCall(name: string, input: unknown) {
  if (!isRecord(input)) throw new ModelError('Qwen3.5 tool arguments must be an object.')
  const parameters = Object.entries(input).map(([key, value]) => {
    const text = typeof value === 'string' ? value : JSON.stringify(value)
    return `<parameter=${key}>${text}</parameter>`
  })
  return `<tool_call><function=${name}>${parameters.join('')}</function></tool_call>`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
