import type { ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import { ModelError, type ToolSpec } from '../types.js'
import { writeValue } from './value.js'
import { gemmaSchema } from './schema.js'

export type GemmaMessage = ChatCompletionMessageParam & {
  reasoning?: string
  response?: unknown
  context?: string
}

export function gemmaPrompt(history: GemmaMessage[], tools: ToolSpec[], thinking = false) {
  const first = history[0]
  const system = first?.role === 'system' ? first.content.trim() : ''
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
    if (
      'name' in message &&
      message.name !== undefined &&
      (message.role !== 'assistant' || message.name !== 'model')
    )
      throw new ModelError('Gemma protocol mode cannot preserve message names.')
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
      prompt += '<|turn>user\n' + message.content.trim() + '<turn|>\n'
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
