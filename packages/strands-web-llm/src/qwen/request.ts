import { isQwen35 } from '../../../../shared/web-llm/models.js'
import { toolPrompt } from '../../../../shared/web-llm/qwen/tools.js'
import { systemPrompt as xmlSystem } from '../../../../shared/web-llm/qwen35/tools.js'
import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../chat/messages.js'
import { generationOptions, hasStructuredResponse } from '../request.js'
import { qwenHistory } from './history.js'
import { qwenToolResponseFormat } from '../../../../shared/web-llm/qwen/constraints.js'
import { qwen25SystemPrompt } from '../../../../shared/web-llm/qwen/history.js'

export function qwenRequest(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
): ChatCompletionRequestStreaming {
  const xml = isQwen35(config.modelId)
  const tools = options.toolSpecs ?? []
  const responseFormat = qwenToolResponseFormat(
    config.modelId,
    config.request ?? {},
    tools,
    options.toolChoice,
    config.thinkingPrefilled,
  )
  if (tools.length && hasStructuredResponse(config))
    throw new ModelError(
      `${xml ? 'Qwen3.5' : 'Qwen'} tool calls cannot use a structured response format.`,
    )
  const request = generationOptions(config)
  if (responseFormat) request.response_format = responseFormat
  delete request.tool_choice
  return {
    ...request,
    messages: xml
      ? qwen35Messages(messages, options)
      : qwenMessages(config.modelId.startsWith('Qwen3-'), messages, options),
  }
}

function qwenMessages(isQwen3: boolean, messages: Message[], options: StreamOptions) {
  return qwenHistory(messages, options, {
    systemPrompt: qwenSystemPrompt(isQwen3, options),
    keepThoughts: isQwen3,
  })
}

export function qwenSystemPrompt(isQwen3: boolean, options: StreamOptions) {
  const system = formatSystemPrompt(options.systemPrompt)
  const prompt = toolPrompt(options.toolSpecs ?? [], options.toolChoice)
  const systemParts = []
  if (system !== undefined) systemParts.push(system)
  else if (!isQwen3) systemParts.push(qwen25SystemPrompt)
  if (prompt) systemParts.push(prompt)
  return systemParts.length ? systemParts.join('\n\n') : undefined
}

function qwen35Messages(messages: Message[], options: StreamOptions) {
  return qwenHistory(messages, options, {
    systemPrompt: qwen35SystemPrompt(options),
    keepThoughts: true,
    xml: true,
  })
}

export function qwen35SystemPrompt(options: StreamOptions) {
  return xmlSystem(
    formatSystemPrompt(options.systemPrompt),
    options.toolSpecs ?? [],
    options.toolChoice,
  )
}
