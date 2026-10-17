import {
  ModelError,
  type CountTokensOptions,
  type Message,
  type StreamOptions,
  type ToolSpec,
} from '@strands-agents/sdk'
import { gemmaHistory } from './history.js'
import { writeValue } from './value.js'
import { gemmaSchema } from './schema.js'

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
) {
  const history = gemmaHistory(messages, options.systemPrompt, options.dynamicTrailingBlocks)
  const first = history[0]
  const system = first?.role === 'system' ? first.content : ''
  if (first?.role === 'system') history.shift()
  let prompt = '<|turn>system\n' + (thinking ? '<|think|>\n' : '') + system
  for (const tool of tools) {
    prompt +=
      '<|tool>declaration:' +
      tool.name +
      writeValue({
        description: tool.description,
        parameters: gemmaSchema(tool.inputSchema ?? { type: 'object', properties: {} }),
      }) +
      '<tool|>'
  }
  prompt += '<turn|>\n'
  const pending = new Map<string, string>()
  let modelOpen = false
  for (const [index, message] of history.entries()) {
    if (message.role === 'tool') {
      const name = pending.get(message.tool_call_id)
      if (!name || !modelOpen)
        throw new ModelError('Gemma received a tool result without a matching call.')
      // Gemma puts JSON object fields directly in the response.
      let response = message.response !== undefined ? message.response : message.content
      if (message.context !== undefined) response = { context: message.context, value: response }
      else if (response === null || typeof response !== 'object' || Array.isArray(response))
        response = { value: response }
      prompt += '<|tool_response>response:' + name + writeValue(response) + '<tool_response|>'
      pending.delete(message.tool_call_id)
      continue
    }
    if (message.role === 'user') {
      if (pending.size) throw new ModelError('Gemma is missing a tool result.')
      if (typeof message.content !== 'string')
        throw new ModelError('Gemma protocol mode supports text and tool results only.')
      if (modelOpen) prompt += '<turn|>\n'
      prompt += '<|turn>user\n' + message.content + '<turn|>\n'
      modelOpen = false
      continue
    }
    if (message.role !== 'assistant')
      throw new ModelError('Gemma received an unexpected system message.')
    if (pending.size) throw new ModelError('Gemma is missing a tool result.')
    if (!modelOpen) prompt += '<|turn>model\n'
    modelOpen = true
    if (message.reasoning) prompt += '<|channel>thought\n' + message.reasoning + '\n<channel|>'
    prompt += message.content ?? ''
    for (const call of message.tool_calls ?? []) {
      if (pending.has(call.id)) throw new ModelError('Gemma received a duplicate tool call ID.')
      const input: unknown = JSON.parse(call.function.arguments)
      prompt += '<|tool_call>call:' + call.function.name + writeValue(input) + '<tool_call|>'
      pending.set(call.id, call.function.name)
    }
    if (!message.tool_calls?.length && history[index + 1]?.role !== 'assistant') {
      prompt += '<turn|>\n'
      modelOpen = false
    }
  }
  if (pending.size) throw new ModelError('Gemma is missing a tool result.')
  if (!modelOpen) prompt += '<|turn>model\n'
  // WebLLM's Gemma artifact loader supplies the BOS token itself.
  return prompt
}
