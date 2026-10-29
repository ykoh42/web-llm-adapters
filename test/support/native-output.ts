import type {
  ChatCompletionChunk,
  ChatCompletionRequestBase,
  Completion,
  CompletionCreateParamsBase,
} from '@mlc-ai/web-llm'

/** Check native requests for a tool result, independent of the model's answer. */
export function hasToolResult(
  requests: readonly (ChatCompletionRequestBase | CompletionCreateParamsBase)[],
  value: string,
) {
  if (!value) return false
  return requests.some((request) =>
    'prompt' in request
      ? inTag(request.prompt, '<|tool_response>', '<tool_response|>', value)
      : request.messages.some(
          (message) =>
            typeof message.content === 'string' &&
            (message.role === 'tool'
              ? message.content.includes(value)
              : message.role === 'user' &&
                inTag(message.content, '<tool_response>', '</tool_response>', value)),
        ),
  )
}

function inTag(text: string, open: string, close: string, value: string) {
  let index = 0
  while ((index = text.indexOf(open, index)) !== -1) {
    const start = index + open.length
    const end = text.indexOf(close, start)
    if (end === -1) return false
    const match = text.indexOf(value, start)
    if (match !== -1 && match + value.length <= end) return true
    index = end + close.length
  }
  return false
}

/** Final answer text and thinking from every response, including prompt prefixes. */
export function nativeOutput(
  chunks: readonly (ChatCompletionChunk | Completion)[],
  modelId: string,
  prefix: string | readonly string[] = '',
) {
  if (!chunks.length) throw new Error('No native response was recorded.')
  const turns = new Map<string, string>()
  for (const chunk of chunks) {
    const choice = chunk.choices[0]
    const text = choice ? ('delta' in choice ? (choice.delta.content ?? '') : choice.text) : ''
    turns.set(chunk.id, (turns.get(chunk.id) ?? '') + text)
  }
  const responses = [...turns.values()].map(
    (text, index) => (typeof prefix === 'string' ? prefix : (prefix[index] ?? '')) + text,
  )
  const last = responses.at(-1)!
  const thought = modelId.startsWith('gemma-4-')
    ? /<\|channel>thought\n([\s\S]*?)<channel\|>/g
    : modelId.startsWith('Qwen3')
      ? /<think>([\s\S]*?)<\/think>/g
      : undefined
  if (!thought) return { text: last, thinking: responses.map(() => '') }
  return {
    text: last.replace(thought, ''),
    thinking: responses.map((text) =>
      [...text.matchAll(thought)].map((match) => match[1]).join(''),
    ),
  }
}
