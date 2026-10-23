import { ModelError, type Message, type SystemPrompt, type ToolChoice } from '@strands-agents/sdk'
import type { ContentBlock, ToolResultBlock, ToolResultContent } from '@strands-agents/sdk'
import type {
  ChatCompletionContentPart,
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionToolChoiceOption,
} from '@mlc-ai/web-llm'
import { formatUserContent } from '../../../../shared/web-llm/content.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'

export function formatMessages(
  messages: Message[],
  systemPrompt?: SystemPrompt,
): ChatCompletionMessageParam[] {
  const output: ChatCompletionMessageParam[] = []
  const text = formatSystemPrompt(systemPrompt)
  if (text !== undefined) output.push({ role: 'system', content: text })

  for (const message of messages) {
    if (message.role === 'assistant') output.push(formatAssistant(message.content))
    else output.push(...formatUser(message.content))
  }
  return output
}

export function formatAssistant(content: ContentBlock[]): ChatCompletionAssistantMessageParam {
  rejectSignedToolCalls(content)
  const calls = content.filter((block) => block.type === 'toolUseBlock')
  const text = formatText(content.filter((block) => block.type !== 'toolUseBlock'))
  return {
    role: 'assistant',
    content: text,
    ...(calls.length && {
      tool_calls: calls.map((call) => ({
        id: call.toolUseId,
        type: 'function' as const,
        function: { name: call.name, arguments: stringifyJSON(call.input) },
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

export function formatUser(
  content: ContentBlock[],
  structuredResults = false,
): (ChatCompletionMessageParam & { response?: unknown })[] {
  const output: (ChatCompletionMessageParam & { response?: unknown })[] = []
  const toolImageMessages: ChatCompletionMessageParam[] = []
  // Tool results must precede user text injected into the same Strands message.
  for (const block of content) {
    if (block.type !== 'toolResultBlock') continue
    const images = block.content.filter((part) => part.type === 'imageBlock').map(formatPart)
    const response =
      structuredResults && block.status !== 'error' ? structuredResult(block) : undefined
    const text =
      response !== undefined
        ? ''
        : formatText(block.content.filter((part) => part.type !== 'imageBlock'))
    const result = block.status === 'error' ? (text ? `[ERROR] ${text}` : '[ERROR]') : text
    output.push({
      role: 'tool',
      tool_call_id: block.toolUseId,
      content: result,
      ...(response !== undefined && { response }),
    })
    if (images.length) toolImageMessages.push({ role: 'user', content: images })
  }
  output.push(...toolImageMessages)
  const parts = content
    .filter((block) => block.type !== 'toolResultBlock' && block.type !== 'cachePointBlock')
    .map(formatPart)
  if (parts.length || !output.length)
    output.push({ role: 'user', content: formatUserContent(parts) })
  return output
}

function structuredResult(block: ToolResultBlock): unknown {
  const first = block.content[0]
  if (block.content.length === 1 && first?.type === 'jsonBlock')
    return JSON.parse(stringifyJSON(first.json))
  if (block.content.length < 2 || block.content.every((part) => part.type === 'textBlock'))
    return undefined
  formatText(
    block.content.filter((part) => part.type !== 'jsonBlock' && part.type !== 'imageBlock'),
  )
  const snapshots: unknown[] = JSON.parse(stringifyJSON(block.content.map((part) => part.toJSON())))
  snapshots.forEach((snapshot, index) => {
    if (
      block.content[index]?.type === 'jsonBlock' &&
      (typeof snapshot !== 'object' || snapshot === null || !Object.hasOwn(snapshot, 'json'))
    )
      throw new TypeError('JSON values must be serializable.')
  })
  return snapshots
}

export function hasToolContext(message: Message | undefined, trailingBlocks = 0): boolean {
  if (message?.role !== 'user' || trailingBlocks <= 0) return false
  const before = message.content.slice(0, -trailingBlocks)
  const context = message.content.slice(-trailingBlocks)
  return context.some((block) => block.type === 'textBlock') && isToolReply(before)
}

export function lastQueryIndex(messages: Message[], trailingBlocks = 0): number {
  const lastMessage = messages.at(-1)
  const toolContext = hasToolContext(lastMessage, trailingBlocks)
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!
    if (
      message.role === 'user' &&
      !(toolContext && message === lastMessage) &&
      !isToolReply(message.content)
    )
      return index
  }
  return -1
}

function isToolReply(content: readonly ContentBlock[]): boolean {
  return (
    content.some((block) => block.type === 'toolResultBlock') &&
    content.every((block) => block.type === 'toolResultBlock' || block.type === 'cachePointBlock')
  )
}

export function formatText(blocks: readonly (ContentBlock | ToolResultContent)[]): string {
  return blocks
    .map((block) => {
      if (block.type === 'textBlock') return block.text
      if (block.type === 'jsonBlock') return stringifyJSON(block.json)
      if (block.type === 'cachePointBlock') return ''
      if (block.type === 'reasoningBlock') {
        if (block.signature !== undefined || block.redactedContent !== undefined)
          throw new ModelError('WebLLM cannot use signed or redacted reasoning blocks.')
        return block.text ?? ''
      }
      throw new ModelError(`WebLLM cannot use ${block.type} here.`)
    })
    .join('')
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

export function formatToolChoice(choice: ToolChoice): ChatCompletionToolChoiceOption {
  if ('auto' in choice) return 'auto'
  if ('tool' in choice) return { type: 'function', function: { name: choice.tool.name } }
  throw new ModelError('toolChoice.any needs exactly one tool for this model.')
}

/** Cache hints are harmless; unsupported system blocks must never disappear silently. */
export function formatSystemPrompt(prompt?: SystemPrompt): string | undefined {
  if (prompt === undefined || typeof prompt === 'string') return prompt
  return formatText(prompt)
}
