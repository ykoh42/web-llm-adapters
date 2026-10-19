import { deepCompareStrict } from '@langchain/core/utils/json_schema'
import { AIMessage, type BaseMessage, type ToolCall } from '@langchain/core/messages'
import type { ChatCompletionContentPart, ChatCompletionMessageParam } from '@mlc-ai/web-llm'
import type { GemmaMessage } from '../../../../shared/web-llm/gemma/prompt.js'

export function formatMessages(messages: BaseMessage[]): GemmaMessage[] {
  const output: GemmaMessage[] = []
  for (const message of messages) {
    const role = message.getType()
    if (role === 'system') {
      const content = textContent(message.content)
      const first = output[0]
      if (first?.role === 'system' && output.length === 1) first.content += '\n\n' + content
      else {
        if (output.length) throw new Error('WebLLM needs system messages before the conversation.')
        output.push({ role: 'system', content })
      }
      continue
    }
    if (role === 'human') {
      const parts =
        typeof message.content === 'string' ? message.content : message.content.map(formatPart)
      const content =
        Array.isArray(parts) && parts.every((part) => part.type === 'text')
          ? parts.map((part) => part.text).join('')
          : parts
      output.push({ role: 'user', content })
      continue
    }
    if (role === 'tool') {
      if (!('tool_call_id' in message) || typeof message.tool_call_id !== 'string')
        throw new Error('A tool result needs its tool_call_id.')
      output.push({
        role: 'tool',
        tool_call_id: message.tool_call_id,
        content: textContent(message.content),
      })
      continue
    }
    if (role !== 'ai') throw new Error(`WebLLM cannot use a ${role} message.`)
    if (!AIMessage.isInstance(message)) throw new Error('WebLLM needs an AI message.')
    const ai = message
    const calls = assistantCalls(ai)
    if (ai.invalid_tool_calls?.length) throw new Error('WebLLM cannot replay invalid tool calls.')
    const reasoning = reasoningContent(message)
    const content = textContent(message.content, true)
    output.push({
      role: 'assistant',
      content: content || null,
      ...(reasoning && { reasoning }),
      ...(calls.length && {
        tool_calls: calls.map((call) => {
          if (!call.id) throw new Error('A tool call needs an ID before it can be replayed.')
          return {
            id: call.id,
            type: 'function' as const,
            function: { name: call.name, arguments: JSON.stringify(call.args) },
          }
        }),
      }),
    })
  }
  return output
}

function textContent(content: BaseMessage['content'], allowReasoning = false): string {
  if (typeof content === 'string') return content
  return content
    .map((block) => {
      if (block.type === 'text' && typeof block.text === 'string') return block.text
      if (allowReasoning && ['reasoning', 'tool_use', 'tool_call'].includes(block.type)) return ''
      throw new Error(`WebLLM cannot use ${block.type} in this message.`)
    })
    .join('')
}

function reasoningContent(message: BaseMessage): string {
  const parts: string[] = []
  if (Array.isArray(message.content))
    for (const block of message.content) {
      if (block.type !== 'reasoning') continue
      if (typeof block.reasoning !== 'string' || block.signature !== undefined)
        throw new Error('WebLLM needs unsigned text reasoning.')
      parts.push(block.reasoning)
    }
  const extra = message.additional_kwargs.reasoning_content
  if (!parts.length && extra !== undefined) {
    if (typeof extra !== 'string') throw new Error('WebLLM needs text reasoning_content.')
    return extra
  }
  return parts.join('')
}

function formatPart(value: unknown): ChatCompletionContentPart {
  if (!isRecord(value)) throw new Error('WebLLM needs an object content block.')
  if (value.type === 'text' && typeof value.text === 'string')
    return { type: 'text', text: value.text }
  if (value.type === 'image_url') {
    const image = value.image_url
    const url = typeof image === 'string' ? image : isRecord(image) ? image.url : undefined
    if (typeof url === 'string') return { type: 'image_url', image_url: { url } }
  }
  const mime = value.mimeType ?? value.mime_type
  if (value.type === 'image') {
    if (typeof value.url === 'string') return { type: 'image_url', image_url: { url: value.url } }
    if (typeof value.data === 'string' && typeof mime === 'string')
      return {
        type: 'image_url',
        image_url: { url: `data:${mime};base64,${value.data}` },
      }
    if (typeof value.base64 === 'string' && typeof value.mime_type === 'string')
      return {
        type: 'image_url',
        image_url: { url: `data:${value.mime_type};base64,${value.base64}` },
      }
  }
  if (value.type === 'input_audio' && isRecord(value.input_audio)) {
    if (value.input_audio.format === 'wav' && typeof value.input_audio.data === 'string')
      return { type: 'input_audio', input_audio: { format: 'wav', data: value.input_audio.data } }
  }
  if (value.type === 'audio' && mime === 'audio/wav' && typeof value.data === 'string')
    return { type: 'input_audio', input_audio: { format: 'wav', data: value.data } }
  throw new Error(
    `WebLLM cannot use this ${String(value.type)} block. Use image URLs/base64 or WAV audio.`,
  )
}

export function nativeMessages(messages: GemmaMessage[]): ChatCompletionMessageParam[] {
  return messages.map((message) => {
    const native = { ...message }
    delete native.reasoning
    delete native.response
    delete native.context
    return native
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Normalize tool content without replaying duplicate provider-specific tool blocks. */
function assistantCalls(message: AIMessage): ToolCall[] {
  const calls: ToolCall[] = [...(message.tool_calls ?? [])]
  if (!Array.isArray(message.content)) return calls
  for (const block of message.content) {
    if (block.type !== 'tool_use' && block.type !== 'tool_call') continue
    const args = block.type === 'tool_use' ? block.input : block.args
    if (typeof block.id !== 'string' || typeof block.name !== 'string' || !isRecord(args))
      throw new Error('WebLLM needs a tool name, ID, and object arguments.')
    const previous = calls.find((call) => call.id === block.id)
    if (previous) {
      if (previous.name !== block.name || !deepCompareStrict(previous.args, args))
        throw new Error('WebLLM received conflicting tool call content.')
    } else calls.push({ id: block.id, name: block.name, args, type: 'tool_call' })
  }
  return calls
}
