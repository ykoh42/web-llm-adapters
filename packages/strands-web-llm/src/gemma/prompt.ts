import {
  type CountTokensOptions,
  type Message,
  type StreamOptions,
  type ToolSpec,
} from '@strands-agents/sdk'
import { gemmaHistory } from './history.js'
import { gemmaPrompt } from '../../../../shared/web-llm/gemma/prompt.js'

/** Estimate native declarations instead of counting their much larger SDK JSON form. */
export function promptTokens(options: CountTokensOptions, thinking: boolean) {
  const prompt = toolPrompt([], options.toolSpecs ?? [], options, thinking)
  let special = 0
  const text = prompt.replace(
    /<\|(?:turn|tool|tool_call|tool_response|channel)>|<(?:turn|tool|tool_call|tool_response|channel)\|>|<\|(?:think|")\|>/g,
    () => {
      special++
      return ''
    },
  )
  return special + Math.ceil(text.length / 4)
}

// Tool format: https://huggingface.co/google/gemma-4-E2B-it/blob/main/chat_template.jinja
export function toolPrompt(
  messages: Message[],
  tools: ToolSpec[],
  options: StreamOptions,
  thinking = false,
  preserve = false,
) {
  const history = gemmaHistory(
    messages,
    options.systemPrompt,
    options.dynamicTrailingBlocks,
    preserve,
  )
  return gemmaPrompt(history, tools, thinking)
}
