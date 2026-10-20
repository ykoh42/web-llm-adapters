import {
  AIMessage,
  ToolMessage,
  type BaseMessage,
  type MessageFieldWithRole,
} from '@langchain/core/messages'
import { formatPart } from '../chat/messages.js'

/** Send model inputs without storage escapes or private tool artifacts. */
export function browserMessages(messages: BaseMessage[]): MessageFieldWithRole[] {
  return messages.map((message) => {
    const role = message.type
    if (role !== 'human' && role !== 'ai' && role !== 'system' && role !== 'tool')
      throw new Error(`WebLLM cannot use a ${role} message.`)
    const reasoning = message.additional_kwargs.reasoning_content
    return {
      role,
      content: browserContent(message.content),
      ...(message.name !== undefined && { name: message.name }),
      ...(message.id !== undefined && { id: message.id }),
      ...(reasoning !== undefined && { additional_kwargs: { reasoning_content: reasoning } }),
      ...(AIMessage.isInstance(message) && {
        tool_calls: message.tool_calls,
        invalid_tool_calls: message.invalid_tool_calls,
      }),
      ...(ToolMessage.isInstance(message) && {
        tool_call_id: message.tool_call_id,
        ...(message.status !== undefined && { status: message.status }),
      }),
    }
  })
}

function browserContent(content: BaseMessage['content']) {
  if (typeof content === 'string') return content
  return content.map((part) => {
    if ((part.type === 'image' || part.type === 'audio') && part.data instanceof Uint8Array)
      return { ...formatPart(part) }
    return part
  })
}
