import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { openEngine } from '../../src/engine/load.js'

const mocks = vi.hoisted(() => ({
  register: vi.fn(),
  reload: vi.fn(),
  unload: vi.fn(),
}))

vi.mock('../../src/engine/service-worker.ts?worker&url', () => ({
  default: '/new-service-worker.js',
}))
vi.mock('@mlc-ai/web-llm', () => {
  class Engine {
    reload = mocks.reload
    unload = mocks.unload
  }
  return {
    MLCEngine: Engine,
    WebWorkerMLCEngine: Engine,
    ServiceWorkerMLCEngine: Engine,
    ExtensionServiceWorkerMLCEngine: class {},
  }
})

let workers: EventTarget & {
  controller: { scriptURL: string } | null
  register: typeof mocks.register
}
const scriptURL = 'http://localhost/new-service-worker.js'

beforeEach(() => {
  vi.resetAllMocks()
  workers = Object.assign(new EventTarget(), { controller: null, register: mocks.register })
  vi.stubGlobal('navigator', { serviceWorker: workers })
  vi.stubGlobal('location', { href: 'http://localhost/' })
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function load() {
  return openEngine('service-worker', 'loaded-model', 8192, { model_list: [] }, () => {})
}

it('uses an already controlling worker when its script matches', async () => {
  workers.controller = { scriptURL }
  const loaded = await load()
  expect(mocks.register).toHaveBeenCalledExactlyOnceWith('/new-service-worker.js', {
    type: 'module',
    scope: '/',
  })
  expect(mocks.reload).toHaveBeenCalledExactlyOnceWith('loaded-model', {
    context_window_size: 8192,
  })
  await loaded.close()
  expect(mocks.unload).toHaveBeenCalledOnce()
})

it('waits for the registered worker instead of sending requests to an older controller', async () => {
  workers.controller = { scriptURL: 'http://localhost/old-service-worker.js' }
  const loading = load()
  try {
    await vi.waitFor(() => expect(mocks.register).toHaveBeenCalledOnce())
    expect(mocks.reload).not.toHaveBeenCalled()
    workers.dispatchEvent(new Event('controllerchange'))
    await Promise.resolve()
    expect(mocks.reload).not.toHaveBeenCalled()
    workers.controller = { scriptURL }
    workers.dispatchEvent(new Event('controllerchange'))
    const loaded = await loading
    expect(mocks.reload).toHaveBeenCalledOnce()
    await loaded.close()
  } finally {
    workers.controller = { scriptURL }
    workers.dispatchEvent(new Event('controllerchange'))
    await loading
  }
})

it('times out and removes its listener if the registered worker never takes control', async () => {
  vi.useFakeTimers()
  workers.controller = { scriptURL: 'http://localhost/old-service-worker.js' }
  const removed = vi.spyOn(workers, 'removeEventListener')
  const failed = load().catch((error: unknown) => error)
  await vi.waitFor(() => expect(vi.getTimerCount()).toBe(1))
  await vi.advanceTimersByTimeAsync(15_000)
  expect(await failed).toHaveProperty(
    'message',
    'Service worker did not take control. Reload the page and try again.',
  )
  expect(removed).toHaveBeenCalledWith('controllerchange', expect.any(Function))
  expect(mocks.reload).not.toHaveBeenCalled()
})
