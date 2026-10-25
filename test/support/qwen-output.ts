import type { ChatCompletionChunk, Completion } from '@mlc-ai/web-llm'

/** Restore the prompt's opening marker for independent, full-frame assertions. */
export function qwenTurns(
  chunks: readonly (ChatCompletionChunk | Completion)[],
  prefilled: boolean,
) {
  const turns = new Map<string, string>()
  for (const chunk of chunks) {
    const choice = chunk.choices[0]
    if (!choice || !('delta' in choice)) continue
    turns.set(chunk.id, (turns.get(chunk.id) ?? '') + (choice.delta.content ?? ''))
  }
  return [...turns.values()].map((text) => (prefilled ? '<think>' + text : text))
}
