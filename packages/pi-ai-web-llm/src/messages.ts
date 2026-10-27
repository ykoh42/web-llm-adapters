import type {
  AssistantMessage,
  ImageContent,
  Model,
  Api,
  TranscriptContext,
  Tool,
  ToolCall,
} from '@earendil-works/pi-ai'
import {
  collapseSystemMessages,
  getInitialSystemMessage,
} from '@earendil-works/pi-ai/utils/transcript'
import { getSystemMessageText } from '@earendil-works/pi-ai/utils/text'
import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'
import { formatUserContent } from '../../../shared/web-llm/content.js'
import { stringifyJSON } from '../../../shared/web-llm/json.js'
import type { ProtocolMessage, ToolSpec } from '../../../shared/web-llm/types.js'

export function formatContext(context: TranscriptContext, model: Model<Api>) {
  const history: ProtocolMessage[] = []
  const pending = new Map<string, string>()
  const transcript = collapseSystemMessages(context)
  const tools = formatTools(getInitialSystemMessage(transcript.messages)?.toolsAdded ?? [])
  let toolImages: ChatCompletionContentPart[] = []
  const flushImages = () => {
    if (!toolImages.length) return
    history.push({ role: 'user', content: toolImages })
    toolImages = []
  }
  for (const message of transcript.messages) {
    if (
      message.role === 'assistant' &&
      (message.stopReason === 'error' || message.stopReason === 'aborted')
    )
      continue
    if (message.role !== 'toolResult') {
      if (pending.size) throw new Error('WebLLM is missing a tool result.')
      flushImages()
    }
    if (message.role === 'system') {
      history.push({
        role: 'system',
        content: getSystemMessageText(message),
      })
    } else if (message.role === 'user') {
      history.push({
        role: 'user',
        content:
          typeof message.content === 'string'
            ? message.content
            : formatUserContent(
                message.content.map((part) =>
                  part.type === 'text' ? { type: 'text', text: part.text } : imagePart(part, model),
                ),
              ),
      })
    } else if (message.role === 'toolResult') {
      if (pending.get(message.toolCallId) !== message.toolName)
        throw new Error('A tool result needs a matching tool call ID and name.')
      pending.delete(message.toolCallId)
      let text = ''
      for (const part of message.content) {
        if (part.type === 'text') text += part.text
        else if (part.type === 'image') toolImages.push(imagePart(part, model))
      }
      history.push({
        role: 'tool',
        tool_call_id: message.toolCallId,
        content: (message.isError ? '[ERROR] ' : '') + text,
      })
    } else {
      history.push(formatAssistant(message, model, pending))
    }
  }
  if (pending.size) throw new Error('WebLLM is missing a tool result.')
  flushImages()
  return { history, tools }
}

function imagePart(image: ImageContent, model: Model<Api>): ChatCompletionContentPart {
  if (!model.input.includes('image'))
    throw new Error('This WebLLM model is configured for text only.')
  return { type: 'image_url', image_url: { url: `data:${image.mimeType};base64,${image.data}` } }
}

function formatTools(tools: Tool[]): ToolSpec[] {
  return tools.map((tool) => {
    const constraint = tool.constrainedSampling
    if (constraint && (constraint.type === 'grammar' || constraint.strict === 'require'))
      throw new Error('WebLLM cannot guarantee constrained tool schemas.')
    return {
      name: tool.name,
      description: tool.description,
      inputSchema: JSON.parse(stringifyJSON(tool.parameters)),
    }
  })
}

function formatAssistant(
  message: AssistantMessage,
  model: Model<Api>,
  pending: Map<string, string>,
): ProtocolMessage {
  const toolCalls = []
  let text = ''
  let reasoning = ''
  const sameModel =
    message.api === model.api && message.provider === model.provider && message.model === model.id
  for (const part of message.content) {
    switch (part.type) {
      case 'text':
        text += part.text
        break
      case 'thinking':
        if (sameModel && !part.redacted) reasoning += part.thinking
        break
      case 'toolCall':
        toolCalls.push(formatCall(part, pending))
        break
    }
  }
  return {
    role: 'assistant',
    content: text,
    ...(reasoning && { reasoning }),
    ...(toolCalls.length && { tool_calls: toolCalls }),
  }
}

function formatCall(call: ToolCall, pending: Map<string, string>) {
  if (call.argumentsError || call.namespace)
    throw new Error('WebLLM cannot replay invalid or namespaced tool calls.')
  if (pending.has(call.id)) throw new Error('WebLLM received a duplicate tool call ID.')
  pending.set(call.id, call.name)
  return {
    id: call.id,
    type: 'function' as const,
    function: { name: call.name, arguments: stringifyJSON(call.arguments) },
  }
}
