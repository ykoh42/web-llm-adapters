import { isQwen } from '../../../../shared/web-llm/qwen/models.js'
import { toolPrompt } from '../../../../shared/web-llm/qwen/tools.js'
import { ModelError, type Message, type StreamOptions } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../chat/messages.js'
import { generationOptions, hasStructuredResponse } from '../request.js'
import { qwenHistory } from './history.js'

const qwen25SystemPrompt = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

export function usesQwenProtocol(config: WebLLMModelConfig) {
  return isQwen(config.modelId)
}

export function qwenRequest(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
): ChatCompletionRequestStreaming {
  const tools = options.toolSpecs ?? []
  if (tools.length && hasStructuredResponse(config))
    throw new ModelError('Qwen tool calls cannot use a structured response format.')
  const request = generationOptions(config)
  delete request.tool_choice
  const history = qwenMessages(config.modelId.startsWith('Qwen3-'), messages, options)
  return { ...request, messages: history }
}

function qwenMessages(isQwen3: boolean, messages: Message[], options: StreamOptions) {
  const system = formatSystemPrompt(options.systemPrompt)
  const systemParts = []
  if (system !== undefined) systemParts.push(system)
  else if (!isQwen3) systemParts.push(qwen25SystemPrompt)
  const prompt = toolPrompt(options.toolSpecs ?? [], options.toolChoice)
  if (prompt) systemParts.push(prompt)
  return qwenHistory(messages, options, {
    systemPrompt: systemParts.length ? systemParts.join('\n\n') : undefined,
    keepThoughts: isQwen3,
  })
}
