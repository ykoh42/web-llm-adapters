import {
  ModelError,
  type Message,
  type StreamOptions,
  type ToolChoice,
  type ToolSpec,
} from '@strands-agents/sdk'
import type { CompletionCreateParamsStreaming } from '@mlc-ai/web-llm'
import { isTextFormat } from '../../../../shared/web-llm/response.js'
import type { WebLLMModelConfig } from '../types.js'
import { generationOptions } from '../request.js'
import { toolPrompt } from './prompt.js'
import { preserveThinking } from '../../../../shared/web-llm/request.js'
import { singleToolChoice } from '../../../../shared/web-llm/stream/choice.js'

export function isGemma(modelId: string) {
  return /^gemma-4-/i.test(modelId)
}

export function usesGemmaProtocol(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
) {
  if (!isGemma(config.modelId)) return false
  if (options.toolSpecs?.length || config.request?.extra_body?.enable_thinking === true) return true
  return messages.some((message) =>
    message.content.some(
      (block) =>
        block.type === 'reasoningBlock' ||
        block.type === 'toolUseBlock' ||
        block.type === 'toolResultBlock',
    ),
  )
}

export function gemmaRequest(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
) {
  const { response_format, extra_body, ...sampling } = generationOptions(config)
  // Tool choice is handled by the adapter; raw completions have no tool field.
  delete sampling.tool_choice
  if (!isTextFormat(response_format) || extra_body?.resumable?.enabled || sampling.ignore_eos) {
    throw new ModelError(
      'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
    )
  }
  const thinking = extra_body?.enable_thinking === true
  const preserve = preserveThinking(config.modelId, config.request ?? {})
  const tools = options.toolSpecs ?? []
  let prefix = toolPrefix(tools, options.toolChoice)
  const prompt = toolPrompt(messages, tools, options, thinking, preserve)
  if (thinking && !prefix && prompt.endsWith('<tool_response|>')) prefix = '<|channel>thought\n'
  const request: CompletionCreateParamsStreaming = {
    ...sampling,
    prompt: prompt + prefix,
    ...(extra_body?.enable_latency_breakdown !== undefined && {
      extra_body: { enable_latency_breakdown: extra_body.enable_latency_breakdown },
    }),
  }
  return { request, prefix }
}

function toolPrefix(tools: ToolSpec[], choice?: ToolChoice) {
  if (!choice) return ''
  choice = singleToolChoice(choice, tools)
  if ('any' in choice) throw new ModelError('Gemma toolChoice.any needs exactly one tool.')
  if (!('tool' in choice)) return ''

  const name = choice.tool.name
  if (!tools.some((tool) => tool.name === name))
    throw new ModelError('The chosen Gemma tool is not registered.')
  return '<|tool_call>call:' + name + '{'
}
