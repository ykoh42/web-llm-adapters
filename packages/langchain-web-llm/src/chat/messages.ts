import { formatPart } from './content.js'
import { deepCompareStrict } from '@langchain/core/utils/json_schema'
import {
  AIMessage,
  ChatMessage,
  coerceMessageLikeToMessage,
  type BaseMessage,
  type ContentBlock,
  type ToolCall,
} from '@langchain/core/messages'
import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'
import type { ProtocolMessage } from '../../../../shared/web-llm/types.js'
import { formatUserContent } from '../../../../shared/web-llm/content.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'

export function formatMessages(messages: BaseMessage[]): ProtocolMessage[] {
  const output: ProtocolMessage[] = []
  const toolNames = new Map<string, string>()
  let toolImages: ChatCompletionContentPart[] = []
  function flushImages() {
    if (!toolImages.length) return
    output.push({ role: 'user', content: toolImages })
    toolImages = []
  }
  for (const input of messages) {
    const message = input.type === 'generic' ? standardMessage(input) : input
    const role = message.type
    // Keep all tool results together before adding their images as a user turn.
    if (role !== 'tool') flushImages()
    if (role === 'system') {
      if (message.name !== undefined)
        throw new Error('WebLLM system messages cannot preserve names.')
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
      const content =
        typeof message.content === 'string'
          ? message.content
          : formatUserContent(message.content.map((part) => formatPart(part)))
      output.push({
        role: 'user',
        content,
        ...(message.name !== undefined && { name: message.name }),
      })
      continue
    }
    if (role === 'tool') {
      if (!('tool_call_id' in message) || typeof message.tool_call_id !== 'string')
        throw new Error('A tool result needs its tool_call_id.')
      if (message.name !== undefined && toolNames.get(message.tool_call_id) !== message.name)
        throw new Error('A named tool result needs a matching assistant tool call.')
      let content = ''
      if (typeof message.content === 'string') content = message.content
      else
        for (const block of message.content) {
          if (block.type === 'image' || block.type === 'image_url')
            toolImages.push(formatPart(block))
          else content += textBlock(block)
        }
      if ('status' in message && message.status === 'error')
        content = content ? `[ERROR] ${content}` : '[ERROR]'
      output.push({
        role: 'tool',
        tool_call_id: message.tool_call_id,
        content,
      })
      continue
    }
    if (role !== 'ai') throw new Error(`WebLLM cannot use a ${role} message.`)
    if (!AIMessage.isInstance(message)) throw new Error('WebLLM needs an AI message.')
    const calls = assistantCalls(message)
    if (message.invalid_tool_calls?.length)
      throw new Error('WebLLM cannot replay invalid tool calls.')
    const toolCalls = calls.map((call) => {
      if (!call.id) throw new Error('A tool call needs an ID before it can be replayed.')
      toolNames.set(call.id, call.name)
      return {
        id: call.id,
        type: 'function' as const,
        function: { name: call.name, arguments: stringifyJSON(call.args) },
      }
    })
    const reasoning = reasoningContent(message)
    const content = textContent(message.content, true)
    output.push({
      role: 'assistant',
      content,
      ...(message.name !== undefined && { name: message.name }),
      ...(reasoning && { reasoning }),
      ...(toolCalls.length && { tool_calls: toolCalls }),
    })
  }
  flushImages()
  return output
}

/** Use LangChain's standard classes for supported ChatMessage roles. */
export function standardMessage(message: BaseMessage): BaseMessage {
  if (!ChatMessage.isInstance(message)) return message
  const role = message.role
  if (
    role !== 'user' &&
    role !== 'human' &&
    role !== 'assistant' &&
    role !== 'ai' &&
    role !== 'system'
  )
    return message
  return coerceMessageLikeToMessage({
    role,
    content: Array.isArray(message.content) ? message.content.slice() : message.content,
    ...(message.name !== undefined && { name: message.name }),
    ...(message.id !== undefined && { id: message.id }),
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
  })
}

function textContent(content: BaseMessage['content'], allowReasoning = false): string {
  if (typeof content === 'string') return content
  return content.map((block) => textBlock(block, allowReasoning)).join('')
}

function textBlock(block: ContentBlock, allowReasoning = false): string {
  if (block.type === 'text' && typeof block.text === 'string') return block.text
  if (allowReasoning && ['reasoning', 'tool_use', 'tool_call'].includes(block.type)) return ''
  throw new Error(`WebLLM cannot use ${block.type} in this message.`)
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

export { nativeMessages } from '../../../../shared/web-llm/history.js'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Normalize tool content without replaying duplicate provider-specific tool blocks. */
function assistantCalls(message: AIMessage): ToolCall[] {
  const calls: ToolCall[] = [...(message.tool_calls ?? [])]
  if (!Array.isArray(message.content)) return calls
  let byId: Map<string | undefined, ToolCall> | undefined
  for (const block of message.content) {
    if (block.type !== 'tool_use' && block.type !== 'tool_call') continue
    const args = block.type === 'tool_use' ? block.input : block.args
    if (typeof block.id !== 'string' || typeof block.name !== 'string' || !isRecord(args))
      throw new Error('WebLLM needs a tool name, ID, and object arguments.')
    byId ??= indexCalls(calls)
    const previous = byId.get(block.id)
    if (previous) {
      if (previous.name !== block.name || !deepCompareStrict(previous.args, args))
        throw new Error('WebLLM received conflicting tool call content.')
    } else {
      const call: ToolCall = { id: block.id, name: block.name, args, type: 'tool_call' }
      calls.push(call)
      byId.set(block.id, call)
    }
  }
  return calls
}

function indexCalls(calls: ToolCall[]) {
  const byId = new Map<string | undefined, ToolCall>()
  // Keep the same first match when caller-provided IDs repeat.
  for (const call of calls) if (!byId.has(call.id)) byId.set(call.id, call)
  return byId
}
