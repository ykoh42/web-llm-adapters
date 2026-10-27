import {
  ModelError,
  type Message,
  type StreamOptions,
  type ToolChoice,
  type ToolSpec,
} from '@strands-agents/sdk'
import { isTextFormat } from '../../../../shared/web-llm/response.js'
import type { WebLLMModelConfig } from '../types.js'
import { generationOptions } from '../request.js'
import { toolPrompt } from './prompt.js'
import { preserveThinking } from '../../../../shared/web-llm/request.js'
import { singleToolChoice } from '../../../../shared/web-llm/stream/choice.js'
import { qwenToolResponseFormat } from '../../../../shared/web-llm/qwen/constraints.js'
import { gemmaCompletion } from '../../../../shared/web-llm/gemma/request.js'
import { isGemma, protocolModelId } from '../../../../shared/web-llm/models.js'

export function usesGemmaProtocol(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
) {
  if (!isGemma(protocolModelId(config.modelId, config.sourceModelId))) return false
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
  const sampling = generationOptions(config)
  const { response_format, extra_body } = sampling
  // Tool choice is handled by the adapter; raw completions have no tool field.
  delete sampling.tool_choice
  if (!isTextFormat(response_format) || extra_body?.resumable?.enabled || sampling.ignore_eos) {
    throw new ModelError(
      'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
    )
  }
  const thinking = extra_body?.enable_thinking === true
  const source = protocolModelId(config.modelId, config.sourceModelId)
  const preserve = preserveThinking(source, config.request ?? {})
  const tools = options.toolSpecs ?? []
  qwenToolResponseFormat(source, config.request ?? {}, tools, options.toolChoice)
  const prefix = toolPrefix(tools, options.toolChoice)
  const prompt = toolPrompt(messages, tools, options, thinking, preserve)
  const completion = gemmaCompletion(sampling, prompt, prefix)
  return { request: completion.createRequest(), prefix: completion.prefix }
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
