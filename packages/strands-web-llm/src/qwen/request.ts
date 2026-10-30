import { isQwen35 } from '../../../../shared/web-llm/models.js'
import { systemPrompt as xmlSystem } from '../../../../shared/web-llm/qwen35/tools.js'
import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../chat/messages.js'
import { generationOptions, hasStructuredResponse } from '../request.js'
import { qwenHistory } from './history.js'
import { qwenToolResponseFormat } from '../../../../shared/web-llm/qwen/constraints.js'
import {
  qwen25SystemPrompt,
  qwenSystemPrompt as formatQwenSystem,
} from '../../../../shared/web-llm/qwen/history.js'

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
  const isQwen3 = !xml && config.modelId.startsWith('Qwen3-')
  return {
    ...request,
    messages: qwenHistory(messages, options, {
      systemPrompt: xml ? qwen35SystemPrompt(options) : qwenSystemPrompt(isQwen3, options),
      keepThoughts: xml || isQwen3,
      xml,
    }),
  }
}

export function qwenSystemPrompt(isQwen3: boolean, options: StreamOptions) {
  const system = formatSystemPrompt(options.systemPrompt)
  const initial = system ?? (isQwen3 ? undefined : qwen25SystemPrompt)
  return formatQwenSystem(initial, options.toolSpecs ?? [], options.toolChoice)
}

export function qwen35SystemPrompt(options: StreamOptions) {
  return xmlSystem(
    formatSystemPrompt(options.systemPrompt),
    options.toolSpecs ?? [],
    options.toolChoice,
  )
}
