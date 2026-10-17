import { prebuiltAppConfig, type AppConfig } from '@mlc-ai/web-llm'
import { expect, it, vi } from 'vitest'
import manifest from 'strands-web-llm/runtime/manifest.json'
import { runtimeConfig } from '../src/runtime.js'

it('selects the packaged runtime without changing models or other settings', () => {
  vi.stubGlobal('window', { location: { href: 'https://example.com/app/' } })
  try {
    const original: AppConfig = { ...prebuiltAppConfig, cacheBackend: 'opfs' }
    const snapshot = structuredClone(original)
    const config = runtimeConfig(original)
    expect(original).toEqual(snapshot)
    expect(config.cacheBackend).toBe('opfs')
    for (const [index, model] of config.model_list.entries()) {
      const previous = original.model_list[index]!
      if (model.model_id !== manifest.model_id) {
        expect(model).toBe(previous)
        continue
      }
      expect(new URL(model.model_lib).origin).toBe('https://example.com')
      expect(model.model_lib).toContain('gemma-4-e2b.wasm')
      expect(model.integrity?.model_lib).toBe(manifest.integrity)
      expect({ ...model, model_lib: previous.model_lib, integrity: previous.integrity }).toEqual({
        ...previous,
        integrity: previous.integrity,
      })
    }
  } finally {
    vi.unstubAllGlobals()
  }
})
