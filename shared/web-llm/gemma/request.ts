import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import type { ToolSpec, ToolChoice } from '../types.js'
import { isTextFormat } from '../response.js'
import { gemmaPrompt, type GemmaMessage } from './prompt.js'

export function gemmaRequest(
  history: GemmaMessage[],
  tools: ToolSpec[],
  choice: ToolChoice | 'none' | undefined,
  sampling: Omit<ChatCompletionRequestStreaming, 'messages'>,
) {
  const thinking = sampling.extra_body?.enable_thinking === true
  if (
    !isTextFormat(sampling.response_format) ||
    sampling.extra_body?.resumable?.enabled ||
    sampling.ignore_eos
  )
    throw new Error(
      'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
    )
  const { extra_body, ...raw } = sampling
  delete raw.response_format
  let prompt = gemmaPrompt(history, tools, thinking)
  let prefix = ''
  if (choice && choice !== 'none' && 'any' in choice)
    throw new Error('Gemma does not support required tool choice. Choose auto or a named tool.')
  if (choice && choice !== 'none' && 'tool' in choice) {
    if (!tools.some((tool) => tool.name === choice.tool.name))
      throw new Error('The chosen Gemma tool is not registered.')
    prefix = '<|tool_call>call:' + choice.tool.name + '{'
  }
  if (thinking && !prefix && prompt.endsWith('<tool_response|>')) prefix = '<|channel>thought\n'
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
