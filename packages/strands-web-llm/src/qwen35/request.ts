import { isQwen35 } from '../../../../shared/web-llm/qwen35/models.js'
import { toolPrompt, formatCall } from '../../../../shared/web-llm/qwen35/tools.js'
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
  formatUser,
} from '../chat/messages.js'
import { generationOptions, hasStructuredResponse } from '../request.js'
import { appendQwenTurns, lastQwenQuery } from '../qwen/history.js'

export function usesQwen35Protocol(config: WebLLMModelConfig) {
  return isQwen35(config.modelId)
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
