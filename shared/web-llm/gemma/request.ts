import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { ProtocolMessage, ToolSpec, ToolChoice } from '../types.js'
import { isTextFormat } from '../response.js'
import { gemmaPrompt } from './prompt.js'
import { singleToolChoice } from '../stream/choice.js'

export function gemmaRequest(
  history: ProtocolMessage[],
  tools: ToolSpec[],
  choice: ToolChoice | 'none' | undefined,
  sampling: Omit<ChatCompletionRequestStreaming, 'messages'>,
) {
  if (choice && choice !== 'none') choice = singleToolChoice(choice, tools)
  const thinking = sampling.extra_body?.enable_thinking === true
  checkGemmaRequest(sampling)
  const prompt = gemmaPrompt(history, tools, thinking)
  let prefix = ''
  if (choice && choice !== 'none' && 'any' in choice)
    throw new Error('Gemma required tool choice needs exactly one tool.')
  if (choice && choice !== 'none' && 'tool' in choice) {
    if (!tools.some((tool) => tool.name === choice.tool.name))
      throw new Error('The chosen Gemma tool is not registered.')
    prefix = '<|tool_call>call:' + choice.tool.name + '{'
  }
  return gemmaCompletion(sampling, prompt, prefix)
}

export function gemmaCompletion(
  sampling: Omit<ChatCompletionRequestStreaming, 'messages'>,
  prompt: string,
  prefix: string,
) {
  const { extra_body, ...raw } = sampling
  delete raw.response_format
  if (extra_body?.enable_thinking === true && !prefix && prompt.endsWith('<tool_response|>'))
    prefix = '<|channel>thought\n'
  prompt += prefix
  return {
    prefix,
    createRequest: () => ({
      ...raw,
      prompt,
      ...(extra_body?.enable_latency_breakdown !== undefined && {
        extra_body: { enable_latency_breakdown: extra_body.enable_latency_breakdown },
      }),
    }),
  }
}

export function checkGemmaRequest(
  request: Pick<ChatCompletionRequestStreaming, 'response_format' | 'extra_body' | 'ignore_eos'>,
) {
  if (
    !isTextFormat(request.response_format) ||
    request.extra_body?.resumable?.enabled ||
    request.ignore_eos
  )
    throw new Error(
      'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
    )
}
