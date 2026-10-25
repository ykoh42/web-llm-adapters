import type { ChatOptions } from '@mlc-ai/web-llm'

/** Restore Qwen3.5's official prefix and stop IDs through WebLLM's public API. */
export function chatOptions(modelId: string): ChatOptions {
  return modelId.startsWith('Qwen3.5-')
    ? { conv_config: { role_empty_sep: '\n<think>\n', stop_token_ids: [248044, 248046] } }
    : {}
}
