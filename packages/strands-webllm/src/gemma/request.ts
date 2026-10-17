import {
  ModelError,
  type Message,
  type StreamOptions,
  type ToolChoice,
  type ToolSpec,
} from '@strands-agents/sdk'
import type { CompletionCreateParamsStreaming, ResponseFormat } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig } from '../types.js'
import { generationOptions } from '../request.js'
import { toolPrompt } from './prompt.js'
import { writeKey } from './value.js'

export function isGemma(modelId: string) {
  return /^gemma-4-/i.test(modelId)
}

export function usesGemmaProtocol(
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
) {
  if (!isGemma(config.modelId)) return false
  if (options.toolSpecs?.length || config.request?.extra_body?.enable_thinking != null) return true
  return messages.some((message) =>
    message.content.some((block) => block.type === 'reasoningBlock'),
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
  if (
    (response_format && !isTextFormat(response_format)) ||
    extra_body?.resumable?.enabled ||
    sampling.ignore_eos
  ) {
    throw new ModelError(
      'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
    )
  }
  const thinking = extra_body?.enable_thinking === true
  const tools = options.toolSpecs ?? []
  let prefix = toolPrefix(tools, options.toolChoice)
  const prompt = toolPrompt(messages, tools, options, thinking)
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

/** Plain text is the raw protocol's existing output mode, so no native option is needed. */
function isTextFormat(format: ResponseFormat) {
  return (
    (format.type == null || format.type === 'text') &&
    format.schema == null &&
    format.grammar == null &&
    format.structural_tag == null
  )
}

function toolPrefix(tools: ToolSpec[], choice?: ToolChoice) {
  if (!choice) return ''
  if ('any' in choice)
    throw new ModelError('WebLLM does not support toolChoice.any. Choose auto or a named tool.')
  if (!('tool' in choice)) return ''

  const name = choice.tool.name
  const tool = tools.find((tool) => tool.name === name)
  if (!tool) throw new ModelError('The chosen Gemma tool is not registered.')
  let prefix = '<|tool_call>call:' + name + '{'
  // A named choice fixes the call, not its values. Start its first required field.
  const first = tool.inputSchema?.required?.[0]
  if (!first || !Object.hasOwn(tool.inputSchema?.properties ?? {}, first)) return prefix
  prefix += writeKey(first) + ':'
  const field = tool.inputSchema?.properties?.[first]
  const type = typeof field === 'object' ? field.type : undefined
  if (type === 'string') prefix += '"'
  else if (type === 'array') prefix += '['
  else if (type === 'object') prefix += '{'
  return prefix
}
