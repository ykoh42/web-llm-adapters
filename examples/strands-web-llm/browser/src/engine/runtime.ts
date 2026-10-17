import type { AppConfig } from '@mlc-ai/web-llm'
import runtimeUrl from 'strands-web-llm/runtime/gemma-4-e2b.wasm?url'
import manifest from 'strands-web-llm/runtime/manifest.json'

// Use WebLLM's public model_lib setting; the provider never replaces a caller's engine.
export function runtimeConfig(config: AppConfig): AppConfig {
  return {
    ...config,
    model_list: config.model_list.map((model) =>
      model.model_id === manifest.model_id
        ? {
            ...model,
            model_lib: new URL(runtimeUrl, window.location.href).href,
            integrity: { ...model.integrity, model_lib: manifest.integrity },
          }
        : model,
    ),
  }
}
