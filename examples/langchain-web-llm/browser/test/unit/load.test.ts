import { afterEach, expect, it, vi } from 'vitest'
import { MLCEngine } from '@mlc-ai/web-llm'

afterEach(async () => {
  await window.webllmLangChainBridge?.close()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.resetModules()
})

it.each([false, true])(
  'releases a failed main-thread load (cleanup fails: %s)',
  async (cleanupFails) => {
    vi.stubGlobal('window', {})
    vi.stubGlobal('location', { href: 'http://localhost/' })
    vi.stubGlobal('document', { querySelector: () => ({}) })
    const failure = new Error('Model loading failed')
    const cleanup = new Error('Engine cleanup failed')
    const reload = vi
      .spyOn(MLCEngine.prototype, 'reload')
      .mockRejectedValueOnce(failure)
      .mockResolvedValue(undefined)
    const release = vi.spyOn(MLCEngine.prototype, 'unload').mockResolvedValue(undefined)
    if (cleanupFails) release.mockRejectedValueOnce(cleanup)
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { checks } = await import('../../src/main.js')
    const id = 'Qwen3.5-4B-q4f16_1-MLC'
    await expect(checks.load(id, false)).rejects.toBe(failure)
    expect(release).toHaveBeenCalledOnce()
    expect(() => checks.engine()).toThrow('Load a model first.')
    await checks.load(id, false)
    expect(checks.model().modelId).toBe(id)
    await checks.engine().unload()
    expect(reload.mock.calls).toEqual([[id], [id]])
    expect(release).toHaveBeenCalledTimes(2)
    expect(logged.mock.calls).toEqual(cleanupFails ? [[cleanup]] : [])
  },
)
