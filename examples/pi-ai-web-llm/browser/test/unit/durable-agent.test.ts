import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { openDurable } from '../../src/durable-agent.js'

const { load, unload, storage, harness } = vi.hoisted(() => ({
  load: vi.fn<() => Promise<void>>(),
  unload: vi.fn<() => Promise<void>>(),
  storage: { close: vi.fn<() => Promise<void>>() },
  harness: {
    root: vi.fn<() => Promise<never>>(),
    close: vi.fn<() => Promise<void>>(),
  },
}))
vi.mock('../../src/engine.js', () => ({
  load,
  unload,
  provider: async () => ({ models: {}, model: {} }),
}))
vi.mock('../../src/storage.js', () => ({ openStorage: async () => storage }))
vi.mock('@earendil-works/pi-durable', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@earendil-works/pi-durable')>()),
  Harness: { open: async () => harness },
}))

beforeEach(() => {
  load.mockResolvedValue(undefined)
  unload.mockResolvedValue(undefined)
  storage.close.mockResolvedValue(undefined)
  harness.close.mockResolvedValue(undefined)
})
afterEach(() => {
  vi.resetAllMocks()
  vi.restoreAllMocks()
})

for (const stage of ['load', 'harness'] as const) {
  it.each([false, true])(
    `keeps the ${stage} startup error after cleanup (cleanup fails: %s)`,
    async (cleanupFails) => {
      const failure = new Error('Conversation startup failed')
      const cleanup = new Error('Conversation cleanup failed')
      const engineCleanup = new Error('Engine cleanup failed')
      const owned = stage === 'load' ? storage : harness
      if (stage === 'load') load.mockRejectedValueOnce(failure)
      else harness.root.mockRejectedValueOnce(failure)
      if (cleanupFails) {
        owned.close.mockRejectedValueOnce(cleanup)
        unload.mockRejectedValueOnce(engineCleanup)
      }
      const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
      await expect(
        openDurable(
          'test-model',
          () => {},
          () => {},
        ),
      ).rejects.toBe(failure)
      expect(owned.close).toHaveBeenCalledOnce()
      expect((stage === 'load' ? harness : storage).close).not.toHaveBeenCalled()
      expect(unload).toHaveBeenCalledOnce()
      expect(logged.mock.calls).toEqual(cleanupFails ? [[cleanup], [engineCleanup]] : [])
    },
  )
}
