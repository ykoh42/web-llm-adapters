import { ModelError, type ToolSpec } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import type { ToolDelta, ChunkOptions } from '../stream/chunks.js'
import { readValue } from './value.js'
import { qwenFrames } from '../qwen/frames.js'
import { QwenTags } from './tags.js'
import type { ArgumentSchemaCache } from './schema.js'

/** Qwen3.5 XML can contain several functions and requires the closing tool tag. */
export function qwen35Chunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  options: ChunkOptions<ChatCompletionChunk> & {
    parseProtocol?: boolean
    thinkingPrefilled?: boolean
    thinkingDisabled?: boolean
    stopRequested?: boolean
  } = {},
) {
  const schemas: ArgumentSchemaCache = new WeakMap()
  const tags = new QwenTags(tools, schemas)
  return qwenFrames(
    chunks,
    {
      markers: QwenTags.markers,
      end: (text, limit) => {
        for (const tag of tags.read(text, limit))
          if (tag.marker === '</tool_call>' && !tags.inFunction && !tags.inParameter)
            return tag.index
        return -1
      },
      parse: (text, index) => parseCalls(text, tools, index, schemas),
      unclosedError() {
        return tags.inParameter
          ? 'Qwen3.5 returned a malformed tool parameter.'
          : 'Qwen3.5 ended inside a tool call.'
      },
    },
    { ...options, parseThinking: true, parseTools: tools.length > 0 },
  )
}

function parseCalls(
  source: string,
  tools: ToolSpec[],
  index: number,
  schemas: ArgumentSchemaCache,
): ToolDelta[] {
  const calls: ToolDelta[] = []
  const tags = new QwenTags(tools, schemas)
  let tool: ToolSpec | undefined
  let input: Record<string, unknown> = Object.create(null)
  let parameter: { name: string; start: number } | undefined
  let offset = 0

  function closeParameter(end: number, tool: ToolSpec) {
    if (!parameter) {
      if (source.slice(offset, end).trim())
        throw new ModelError('Qwen3.5 returned a malformed tool parameter.')
      return
    }
    const { name, start } = parameter
    if (Object.hasOwn(input, name)) throw new ModelError(`Qwen3.5 repeated parameter "${name}".`)
    input[name] = readValue(source.slice(start, end), tool, name, schemas)
    parameter = undefined
  }

  for (const tag of tags.read(source)) {
    if (tag.marker === '<function=') {
      if (tool || source.slice(offset, tag.index).trim())
        throw new ModelError('Qwen3.5 returned a malformed tool call.')
      const header = /^<function=([^>]+)>/.exec(source.slice(tag.index))
      if (!header) throw new ModelError('Qwen3.5 returned a malformed tool call.')
      const name = header[1]!
      tool = tools.find((tool) => tool.name === name)
      if (!tool) throw new ModelError(`Qwen3.5 returned unknown tool "${name}".`)
      input = Object.create(null)
      offset = tag.index + header[0].length
      continue
    }
    if (!tool) throw new ModelError('Qwen3.5 returned a malformed tool call.')
    if (tag.marker === '<parameter=') {
      // A new opener may close the previous parameter, as in the recommended parser.
      closeParameter(tag.index, tool)
      const header = /^<parameter=([^>]*)>/.exec(source.slice(tag.index))
      if (!header) throw new ModelError('Qwen3.5 returned a malformed tool parameter.')
      parameter = { name: header[1]!, start: tag.index + header[0].length }
      offset = parameter.start
    } else if (tag.marker === '</parameter>') {
      closeParameter(tag.index, tool)
      offset = tag.end
    } else if (tag.marker === '</function>') {
      if (parameter || source.slice(offset, tag.index).trim())
        throw new ModelError('Qwen3.5 returned a malformed tool parameter.')
      calls.push({
        index: index + calls.length,
        id: crypto.randomUUID(),
        name: tool.name,
        input: JSON.stringify(input),
      })
      tool = undefined
      offset = tag.end
    } else throw new ModelError('Qwen3.5 returned a malformed tool call.')
  }
  if (tool || !calls.length || source.slice(offset).trim())
    throw new ModelError('Qwen3.5 returned a malformed tool call.')
  return calls
}
