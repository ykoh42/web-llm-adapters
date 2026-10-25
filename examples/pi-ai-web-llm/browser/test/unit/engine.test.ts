import { afterEach, expect, it, vi } from 'vitest'
import { MLCEngine } from '@mlc-ai/web-llm'
import { load, provider, unload } from '../../src/engine.js'

afterEach(async () => {
  await unload()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it.each([false, true])(
  'releases a failed main-thread load (cleanup fails: %s)',
  async (cleanupFails) => {
    vi.stubGlobal('location', { href: 'http://localhost/' })
    const failure = new Error('Model loading failed')
    const cleanup = new Error('Engine cleanup failed')
    const reload = vi
      .spyOn(MLCEngine.prototype, 'reload')
      .mockRejectedValueOnce(failure)
      .mockResolvedValue(undefined)
    const release = vi.spyOn(MLCEngine.prototype, 'unload').mockResolvedValue(undefined)
    if (cleanupFails) release.mockRejectedValueOnce(cleanup)
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    const id = 'Qwen3.5-4B-q4f16_1-MLC'
    await expect(load(id, false)).rejects.toBe(failure)
    expect(release).toHaveBeenCalledOnce()
    await expect(provider(false)).rejects.toThrow('Load a model first.')
    await load(id, false)
    expect((await provider(false)).model.id).toBe(id)
    await unload()
    const options = {
      context_window_size: 32768,
      conv_config: { role_empty_sep: '\n<think>\n', stop_token_ids: [248044, 248046] },
    }
    expect(reload.mock.calls).toEqual([
      [id, options],
      [id, options],
    ])
    expect(release).toHaveBeenCalledTimes(2)
    expect(logged.mock.calls).toEqual(cleanupFails ? [[cleanup]] : [])
  },
)
