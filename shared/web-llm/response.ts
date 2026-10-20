import type { ResponseFormat } from '@mlc-ai/web-llm'

export function isTextFormat(format?: ResponseFormat): boolean {
  if (!format) return true
  return (
    (format.type == null || format.type === 'text') &&
    format.schema == null &&
    format.grammar == null &&
    format.structural_tag == null
  )
}

/** WebLLM 0.2.86 passes a missing JSON schema to its grammar compiler. */
export function normalizeResponseFormat(format: ResponseFormat): ResponseFormat {
  if (format.type !== 'json_object' || format.schema !== undefined) return format
  return { ...format, schema: '{}' }
}
