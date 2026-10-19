import { ModelError, type Message, type SystemPrompt, type ToolChoice } from '@strands-agents/sdk'
import type { ContentBlock, ToolResultContent, ToolSpec } from '@strands-agents/sdk'
import type {
  ChatCompletionContentPart,
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from '@mlc-ai/web-llm'

export function formatMessages(
  messages: Message[],
  systemPrompt?: SystemPrompt,
): ChatCompletionMessageParam[] {
  const output: ChatCompletionMessageParam[] = []
  if (systemPrompt !== undefined) {
    const text = formatSystemPrompt(systemPrompt)
    if (text) output.push({ role: 'system', content: text })
  }

  for (const message of messages) {
    if (message.role === 'assistant') {
      const assistant = formatAssistant(message.content)
      if (assistant.content !== null || assistant.tool_calls) output.push(assistant)
    } else output.push(...formatUser(message.content))
  }
  return output
}

export function formatAssistant(content: ContentBlock[]): ChatCompletionAssistantMessageParam {
  rejectSignedToolCalls(content)
  const calls = content.filter((block) => block.type === 'toolUseBlock')
  const text = textOnly(content.filter((block) => block.type !== 'toolUseBlock'))
  return {
    role: 'assistant',
    content: text || (calls.length ? '' : null),
    ...(calls.length && {
      tool_calls: calls.map((call) => ({
        id: call.toolUseId,
        type: 'function' as const,
        function: { name: call.name, arguments: JSON.stringify(call.input) },
      })),
    }),
  }
}

export function rejectSignedToolCalls(content: readonly ContentBlock[]) {
  if (
    content.some((block) => block.type === 'toolUseBlock' && block.reasoningSignature !== undefined)
  )
    throw new ModelError('WebLLM cannot replay tool calls with reasoning signatures.')
}

export function formatUser(content: ContentBlock[]): ChatCompletionMessageParam[] {
  const output: ChatCompletionMessageParam[] = []
  const toolImageMessages: ChatCompletionMessageParam[] = []
  // Tool results must precede user text injected into the same Strands message.
  for (const block of content) {
    if (block.type !== 'toolResultBlock') continue
    const images = block.content.filter((part) => part.type === 'imageBlock').map(formatPart)
    const text = textOnly(block.content.filter((part) => part.type !== 'imageBlock'))
    const result = images.length && !text.trim() ? 'Tool successfully returned an image.' : text
    output.push({
      role: 'tool',
      tool_call_id: block.toolUseId,
      content: block.status === 'error' ? `Tool error: ${result}` : result,
    })
    if (images.length) toolImageMessages.push({ role: 'user', content: images })
  }
  output.push(...toolImageMessages)
  const parts = content
    .filter((block) => block.type !== 'toolResultBlock' && block.type !== 'cachePointBlock')
    .map(formatPart)
  if (!parts.length) return output
  const text = parts
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
  const media = parts.filter((part) => part.type !== 'text')
  output.push({ role: 'user', content: media.length ? [{ type: 'text', text }, ...media] : text })
  return output
}

function textOnly(blocks: readonly (ContentBlock | ToolResultContent)[]): string {
  return blocks
    .map((block) => {
      if (block.type === 'textBlock') return block.text
      if (block.type === 'jsonBlock') return JSON.stringify(block.json)
      if (block.type === 'cachePointBlock') return ''
      if (block.type === 'reasoningBlock') {
        if (block.signature !== undefined || block.redactedContent !== undefined)
          throw new ModelError('WebLLM cannot use signed or redacted reasoning blocks.')
        return block.text ?? ''
      }
      throw new ModelError(`WebLLM cannot use ${block.type} here.`)
    })
    .join('\n')
}

function formatPart(block: ContentBlock): ChatCompletionContentPart {
  if (block.type === 'textBlock') return { type: 'text', text: block.text }
  if (block.type === 'imageBlock') {
    const source = block.toJSON().image.source
    if ('url' in source) {
      return { type: 'image_url', image_url: { url: source.url } }
    }
    if ('bytes' in source) {
      const format = block.format === 'jpg' ? 'jpeg' : block.format
      const url = `data:image/${format};base64,${source.bytes}`
      return { type: 'image_url', image_url: { url } }
    }
  }
  if (block.type === 'audioBlock' && block.format === 'wav') {
    const source = block.toJSON().audio.source
    if ('bytes' in source) {
      return {
        type: 'input_audio',
        input_audio: { format: 'wav', data: source.bytes },
      }
    }
  }
  throw new ModelError(`WebLLM cannot use this ${block.type}. Use image URLs/bytes or WAV bytes.`)
}

export function formatTools(tools: ToolSpec[]): ChatCompletionTool[] {
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: { ...(tool.inputSchema ?? { type: 'object', properties: {} }) },
    },
  }))
}

export function formatToolChoice(choice: ToolChoice): ChatCompletionToolChoiceOption {
  if ('auto' in choice) return 'auto'
  if ('tool' in choice) return { type: 'function', function: { name: choice.tool.name } }
  throw new ModelError('WebLLM does not support toolChoice.any. Choose auto or a named tool.')
}

/** Cache hints are harmless; unsupported system blocks must never disappear silently. */
export function formatSystemPrompt(prompt?: SystemPrompt): string | undefined {
  if (prompt === undefined || typeof prompt === 'string') return prompt
  return textOnly(prompt)
}
