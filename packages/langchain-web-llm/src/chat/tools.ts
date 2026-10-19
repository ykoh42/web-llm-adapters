import type {
  BindToolsInput,
  BaseChatModelCallOptions,
} from '@langchain/core/language_models/chat_models'
import { convertToOpenAITool } from '@langchain/core/utils/function_calling'
import type { JSONSchema, ToolChoice, ToolSpec } from '../../../../shared/web-llm/types.js'

export function formatTools(tools: BindToolsInput[] = []): ToolSpec[] {
  const names = new Set<string>()
  return tools.map((tool) => {
    const native = convertToOpenAITool(tool)
    if (native.type !== 'function') throw new Error('WebLLM supports function tools only.')
    const fn = native.function
    if ('strict' in fn && fn.strict) throw new Error('WebLLM cannot guarantee strict tool schemas.')
    if (typeof fn.name !== 'string' || !fn.name || names.has(fn.name))
      throw new Error('Tools need unique, non-empty names.')
    names.add(fn.name)
    return {
      name: fn.name,
      description: fn.description ?? '',
      inputSchema: structuredClone(
        fn.parameters ?? { type: 'object', properties: {} },
      ) as JSONSchema,
    }
  })
}

export function formatChoice(
  value: BaseChatModelCallOptions['tool_choice'],
): ToolChoice | 'none' | undefined {
  if (value === undefined) return undefined
  if (value === 'none') return 'none'
  if (value === 'auto') return { auto: {} }
  if (value === 'any' || value === 'required') return { any: {} }
  if (typeof value === 'string') return { tool: { name: value } }
  if (value.type === 'function' && value.function && typeof value.function.name === 'string')
    return { tool: { name: value.function.name } }
  throw new Error('Use auto, none, required, or a function name for tool_choice.')
}
