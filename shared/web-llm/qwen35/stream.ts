import { ModelError, type ToolSpec } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import type { ToolDelta, ChunkOptions } from '../stream/chunks.js'
import { readValue } from './value.js'
import { qwenFrames } from '../qwen/frames.js'

/** Qwen3.5 XML can contain several functions and requires the closing tool tag. */
export function qwen35Chunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  options: ChunkOptions<ChatCompletionChunk> & { parseProtocol?: boolean } = {},
) {
  return qwenFrames(
    chunks,
    {
      end: (text) => text.indexOf('</tool_call>'),
      parse: (text, index) => parseCalls(text, tools, index),
      unclosedError: 'Qwen3.5 ended inside a tool call.',
    },
    { ...options, parseThinking: true, parseTools: tools.length > 0 },
  )
}

function parseCalls(source: string, tools: ToolSpec[], index: number): ToolDelta[] {
  const calls: ToolDelta[] = []
  const functions = /<function=([^>\s]+)>([\s\S]*?)<\/function>/g
  for (const match of source.matchAll(functions)) {
    const name = match[1]!
    const tool = tools.find((item) => item.name === name)
    if (!tool) throw new ModelError(`Qwen3.5 returned unknown tool "${name}".`)
    const input = Object.create(null) as Record<string, unknown>
    const body = match[2]!
    const parameters = /<parameter=([^>]*)>([\s\S]*?)(?:<\/parameter>|(?=<parameter=))/g
    const values = [...body.matchAll(parameters)]
    // Like vLLM's Qwen parser, ignore redundant closers outside complete parameters.
    if (body.replace(parameters, '').replaceAll('</parameter>', '').trim())
      throw new ModelError('Qwen3.5 returned a malformed tool parameter.')
    for (const parameter of values) {
      const key = parameter[1]!
      if (Object.hasOwn(input, key)) throw new ModelError(`Qwen3.5 repeated parameter "${key}".`)
      input[key] = readValue(parameter[2]!, tool, key)
    }
    calls.push({
      index: index + calls.length,
      id: crypto.randomUUID(),
      name,
      input: JSON.stringify(input),
    })
  }
  if (!calls.length || source.replace(functions, '').trim())
    throw new ModelError('Qwen3.5 returned a malformed tool call.')
  return calls
}
