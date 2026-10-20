import { isQwen35 } from '../../../../shared/web-llm/qwen35/models.js'
import { toolPrompt } from '../../../../shared/web-llm/qwen35/tools.js'
import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../chat/messages.js'
import { generationOptions, hasStructuredResponse } from '../request.js'
import { qwenHistory } from '../qwen/history.js'

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
    messages: qwen35Messages(messages, options),
  }
}

function qwen35Messages(messages: Message[], options: StreamOptions) {
  const system = formatSystemPrompt(options.systemPrompt)
  const declarations = toolPrompt(options.toolSpecs ?? [], options.toolChoice)
  const prompt = declarations ? [declarations, ...(system ? [system] : [])].join('\n\n') : system
  return qwenHistory(messages, options, {
    systemPrompt: prompt,
    keepThoughts: true,
    xml: true,
  })
}
