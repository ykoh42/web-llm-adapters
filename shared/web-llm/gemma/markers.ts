import { ModelError } from '../types.js'

export const toolStart = '<|tool_call>call:'

export function validateToolName(name: string) {
  if (/[{}]/.test(name) || /<[^>]*\|[^>]*>/.test(name))
    throw new ModelError('Gemma tool names cannot contain braces or control markers.')
}
