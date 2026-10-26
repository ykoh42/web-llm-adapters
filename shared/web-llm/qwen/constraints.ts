import type { ResponseFormat } from '@mlc-ai/web-llm'
import { nativeTools } from '../native.js'
import { ModelError, type ToolChoice, type ToolSpec } from '../types.js'
import type { WebLLMRequest } from '../request.js'
import { supportsConstrainedToolCalls } from './models.js'
import { isQwen35 } from '../qwen35/models.js'
import { xmlToolTag } from '../qwen35/constraints.js'
import { isTextFormat } from '../response.js'

interface ToolTag {
  type: 'tag'
  begin: string
  content: Record<string, unknown>
  end: string
}

export function qwenToolResponseFormat(
  modelId: string,
  request: WebLLMRequest,
  tools: ToolSpec[],
  choice?: ToolChoice | 'none',
  thinkingPrefilled = false,
): ResponseFormat | undefined {
  if (!tools.length || choice === 'none') return

  if (request.constrainedToolCalls !== true) return
  if (!supportsConstrainedToolCalls(modelId))
    throw new ModelError(`Constrained tool calls are not verified for ${modelId}.`)
  if (!isTextFormat(request.response_format))
    throw new ModelError('constrainedToolCalls cannot be combined with request.response_format.')

  const selectedTools =
    choice && 'tool' in choice ? tools.filter((tool) => tool.name === choice.tool.name) : tools
  if (!selectedTools.length) throw new ModelError('The chosen Qwen tool is not registered.')

  const tags = isQwen35(modelId)
    ? selectedTools.map(xmlToolTag)
    : nativeTools(selectedTools).map(
        ({ function: tool }) =>
          ({
            type: 'tag',
            begin: `<tool_call>\n{"name": ${JSON.stringify(tool.name)}, "arguments": `,
            content: { type: 'json_schema', json_schema: tool.parameters },
            end: '}\n</tool_call>',
          }) satisfies ToolTag,
      )
  if (tags.some((tag) => !tag))
    throw new ModelError('Qwen3.5 tool constraints cannot represent this tool schema.')
  const toolTags = tags.filter((tag) => tag !== undefined)
  const parseThinking = !modelId.startsWith('Qwen2.5-')
  const thinking = parseThinking && request.extra_body?.enable_thinking !== false
  const required = Boolean(choice && !('auto' in choice))
  const format = toolFormat(toolTags, required, parseThinking)

  return {
    type: 'structural_tag',
    structural_tag: {
      type: 'structural_tag',
      format:
        thinking && thinkingPrefilled
          ? {
              type: 'sequence',
              elements: [
                {
                  type: 'tag',
                  begin: '',
                  content: pendingText(),
                  end: '</think>',
                },
                format,
              ],
            }
          : format,
    },
  }
}

/** A thinking block alone must never satisfy a required tool choice. */
function toolFormat(tags: ToolTag[], required: boolean, parseThinking: boolean) {
  const thinkingStart = parseThinking ? ['<think>'] : []
  const thoughts: ToolTag[] = parseThinking
    ? [{ type: 'tag', begin: '<think>', content: pendingText(), end: '</think>' }]
    : []
  const suffix = {
    type: 'triggered_tags',
    triggers: [...thinkingStart, '<tool_call>'],
    tags: [...thoughts, ...tags],
    at_least_one: false,
    stop_after_first: false,
  }
  if (!required) return suffix
  const pending = pendingText()
  return {
    type: 'sequence',
    elements: [
      {
        type: 'tag',
        begin: '',
        content: {
          ...pending,
          triggers: [...thinkingStart, ...pending.triggers],
          tags: [...thoughts, ...pending.tags],
        },
        end: '<tool_call>',
      },
      {
        type: 'or',
        elements: tags.map((tag) => ({ ...tag, begin: tag.begin.slice('<tool_call>'.length) })),
      },
      suffix,
    ],
  }
}

/** Legacy stop IDs must not let a stop string end an unfinished thought or required call. */
function pendingText() {
  const stops = ['<|im_end|', '<|endoftext|']
  return {
    type: 'triggered_tags',
    triggers: stops,
    tags: stops.map((begin) => ({
      type: 'tag',
      begin,
      content: { type: 'grammar', grammar: 'root ::= [^><]' },
      end: '',
    })),
    at_least_one: false,
    stop_after_first: false,
  }
}
