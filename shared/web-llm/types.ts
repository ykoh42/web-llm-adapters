import type { JSONSchema7 } from 'json-schema'

export type JSONSchema = JSONSchema7
export interface ToolSpec {
  name: string
  description: string
  inputSchema?: JSONSchema
}
export type ToolChoice =
  { auto: Record<string, never> } | { any: Record<string, never> } | { tool: { name: string } }
export type StopReason = 'endTurn' | 'maxTokens' | 'cancelled' | 'toolUse'

export class ModelError extends Error {
  override name = 'WebLLMError'
}

export type JSONValue =
  string | number | boolean | null | { [key: string]: JSONValue } | JSONValue[]
