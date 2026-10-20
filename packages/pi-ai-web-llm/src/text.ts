import type {
  AssistantMessage,
  AssistantMessageEventStream,
  TextContent,
  ThinkingContent,
} from '@earendil-works/pi-ai'

export function textBlocks(message: AssistantMessage, stream: AssistantMessageEventStream) {
  let active: { block: TextContent | ThinkingContent; index: number } | undefined

  function append(type: 'text' | 'thinking', value: string) {
    if (!value) return
    if (active?.block.type !== type) {
      close()
      const block: TextContent | ThinkingContent =
        type === 'text' ? { type, text: '' } : { type, thinking: '' }
      active = { block, index: message.content.length }
      message.content.push(block)
      stream.push({ type: `${type}_start`, contentIndex: active.index, partial: message })
    }
    const { block, index } = active
    if (block.type === 'text') block.text += value
    else block.thinking += value
    stream.push({ type: `${type}_delta`, contentIndex: index, delta: value, partial: message })
  }

  function close() {
    if (!active) return
    const { block, index } = active
    stream.push({
      type: `${block.type}_end`,
      contentIndex: index,
      content: block.type === 'text' ? block.text : block.thinking,
      partial: message,
    })
    active = undefined
  }

  return { append, close }
}
