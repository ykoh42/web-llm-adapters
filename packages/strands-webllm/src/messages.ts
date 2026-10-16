import { ModelError, type Message, type SystemPrompt, type ToolChoice } from '@strands-agents/sdk'
import type { ContentBlock, ToolResultContent, ToolSpec } from '@strands-agents/sdk'
import type {
  ChatCompletionContentPart,
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
    const text = typeof systemPrompt === 'string' ? systemPrompt : textOnly(systemPrompt)
    if (text) output.push({ role: 'system', content: text })
  }

  for (const message of messages) {
    if (message.role === 'assistant') {
      const calls = message.content.filter((block) => block.type === 'toolUseBlock')
      const text = textOnly(message.content.filter((block) => block.type !== 'toolUseBlock'))
      output.push({
        role: 'assistant',
        content: text || null,
        ...(calls.length && {
          tool_calls: calls.map((call) => ({
            id: call.toolUseId,
            type: 'function' as const,
            function: { name: call.name, arguments: JSON.stringify(call.input) },
          })),
        }),
      })
      continue
    }

    // Tool results must precede user text injected into the same Strands message.
    for (const block of message.content) {
      if (block.type !== 'toolResultBlock') continue
      const text = textOnly(block.content)
      output.push({
        role: 'tool',
        tool_call_id: block.toolUseId,
        content: block.status === 'error' ? `Tool error: ${text}` : text,
      })
    }
    const content = message.content
      .filter((block) => block.type !== 'toolResultBlock' && block.type !== 'cachePointBlock')
      .map(formatPart)
    if (content.length) {
      const text = content
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join('\n')
      const media = content.filter((part) => part.type !== 'text')
      output.push({
        role: 'user',
        content: media.length ? [{ type: 'text', text }, ...media] : text,
      })
    }
  }
  return output
}

function textOnly(blocks: readonly (ContentBlock | ToolResultContent)[]): string {
  return blocks
    .map((block) => {
      if (block.type === 'textBlock') return block.text
      if (block.type === 'jsonBlock') return JSON.stringify(block.json)
      if (block.type === 'cachePointBlock') return ''
      throw new ModelError(`WebLLM cannot use ${block.type} here.`)
    })
    .join('\n')
}

function formatPart(block: ContentBlock): ChatCompletionContentPart {
  if (block.type === 'textBlock') return { type: 'text', text: block.text }
  if (block.type === 'imageBlock') {
    if (block.source.type === 'imageSourceUrl') {
      return { type: 'image_url', image_url: { url: block.source.url } }
    }
    if (block.source.type === 'imageSourceBytes') {
      const format = block.format === 'jpg' ? 'jpeg' : block.format
      const url = `data:image/${format};base64,${base64(block.source.bytes)}`
      return { type: 'image_url', image_url: { url } }
    }
  }
  if (block.type === 'audioBlock' && block.format === 'wav') {
    if (block.source.type === 'audioSourceBytes') {
      return {
        type: 'input_audio',
        input_audio: { format: 'wav', data: base64(block.source.bytes) },
      }
    }
  }
  throw new ModelError(`WebLLM cannot use this ${block.type}. Use image URLs/bytes or WAV bytes.`)
}

function base64(bytes: Uint8Array): string {
  let text = ''
  for (const byte of bytes) text += String.fromCharCode(byte)
  return btoa(text)
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
