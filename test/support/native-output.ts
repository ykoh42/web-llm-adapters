import type { ChatCompletionChunk, Completion } from '@mlc-ai/web-llm'

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
