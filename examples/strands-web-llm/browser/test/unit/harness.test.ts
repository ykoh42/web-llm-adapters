import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  open: vi.fn(),
  closeEngine: vi.fn(),
  closeBridge: vi.fn(),
}))
vi.mock('../../src/engine/load.js', () => ({
  openEngine: mocks.open,
  cacheConfig: async () => ({}),
}))
vi.mock('strands-web-llm', () => ({
  createBrowserBridge: () => ({ close: mocks.closeBridge }),
}))

beforeEach(async () => {
  vi.resetModules()
  vi.resetAllMocks()
  mocks.open.mockResolvedValue({ engine: {}, close: mocks.closeEngine })
  vi.stubGlobal('window', {})
  vi.stubGlobal('document', { querySelector: () => ({ textContent: '' }) })
  await import('../../src/harness/main.js')
})
afterEach(() => vi.unstubAllGlobals())

it.each(['bridge', 'engine'])('can load again after %s cleanup fails', async (failure) => {
  await window.harness.open('loaded-model')
  const error = new Error('Cleanup failed')
  const close = failure === 'bridge' ? mocks.closeBridge : mocks.closeEngine
  close.mockRejectedValueOnce(error)

  await expect(window.harness.close()).rejects.toBe(error)
  expect(mocks.closeBridge).toHaveBeenCalledOnce()
  expect(mocks.closeEngine).toHaveBeenCalledOnce()
  await window.harness.open('next-model')
  expect(mocks.open).toHaveBeenCalledTimes(2)
})

it('waits for engine cleanup before accepting another model', async () => {
  await window.harness.open('loaded-model')
  let finish!: () => void
  mocks.closeEngine.mockReturnValueOnce(new Promise<void>((resolve) => (finish = resolve)))
  const closing = window.harness.close()
  try {
    await vi.waitFor(() => expect(mocks.closeEngine).toHaveBeenCalledOnce())
    await expect(window.harness.open('too-early')).rejects.toThrow('already loaded')
    expect(mocks.open).toHaveBeenCalledOnce()
  } finally {
    finish()
    await closing
  }
  await window.harness.open('next-model')
  expect(mocks.open).toHaveBeenCalledTimes(2)
})
