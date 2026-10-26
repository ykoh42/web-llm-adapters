import { afterEach, expect, it, vi } from 'vitest'
import { MLCEngine, WebWorkerMLCEngine } from '@mlc-ai/web-llm'

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
    const options = {
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

it('terminates a worker when WebLLM worker model loading fails', async () => {
  const worker = { terminate: vi.fn(), postMessage: vi.fn() }
  vi.stubGlobal('window', {})
  vi.stubGlobal('location', { href: 'http://localhost/' })
  vi.stubGlobal('document', { querySelector: () => ({}) })
  vi.stubGlobal(
    'Worker',
    class {
      constructor() {
        return worker
      }
    },
  )
  const failure = new Error('Model loading failed')
  const reload = vi.spyOn(WebWorkerMLCEngine.prototype, 'reload').mockRejectedValueOnce(failure)
  const { checks } = await import('../../src/main.js')

  await expect(checks.load('Qwen3.5-4B-q4f16_1-MLC')).rejects.toBe(failure)
  expect(reload).toHaveBeenCalledOnce()
  expect(worker.terminate).toHaveBeenCalledOnce()
  expect(() => checks.engine()).toThrow('Load a model first.')
})
