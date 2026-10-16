import {
  MLCEngine,
  WebWorkerMLCEngine,
  ServiceWorkerMLCEngine,
  prebuiltAppConfig,
} from '@mlc-ai/web-llm'
import type { AppConfig, InitProgressCallback, MLCEngineInterface } from '@mlc-ai/web-llm'
import serviceWorkerUrl from './service-worker.ts?worker&url'

export async function openEngine(
  kind: string,
  modelId: string,
  contextWindow: number | undefined,
  appConfig: AppConfig,
  initProgressCallback: InitProgressCallback,
) {
  const config = { appConfig, initProgressCallback }
  let worker: Worker | undefined
  let engine: MLCEngineInterface
  switch (kind) {
    case 'worker':
      worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      engine = new WebWorkerMLCEngine(worker, config)
      break
    case 'service-worker':
      await navigator.serviceWorker.register(serviceWorkerUrl, { type: 'module', scope: '/' })
      await waitForController()
      engine = new ServiceWorkerMLCEngine(config)
      break
    case 'main':
      engine = new MLCEngine(config)
      break
    default:
      throw new Error('Choose Main thread, Web Worker, or Service Worker.')
  }
  const close = async () => {
    try {
      await engine.unload()
    } finally {
      worker?.terminate()
    }
  }
  try {
    await engine.reload(modelId, contextWindow ? { context_window_size: contextWindow } : undefined)
    return { engine, close }
  } catch (error) {
    await close().catch((cleanupError: unknown) => console.error(cleanupError))
    throw error
  }
}

function waitForController(): Promise<void> {
  if (navigator.serviceWorker.controller) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const ready = () => {
      clearTimeout(timer)
      navigator.serviceWorker.removeEventListener('controllerchange', ready)
      resolve()
    }
    const timer = setTimeout(() => {
      navigator.serviceWorker.removeEventListener('controllerchange', ready)
      reject(new Error('Service worker did not take control. Reload the page and try again.'))
    }, 15_000)
    navigator.serviceWorker.addEventListener('controllerchange', ready)
    if (navigator.serviceWorker.controller) ready()
  })
}

export function cacheConfig(backend: string): AppConfig {
  if (backend !== 'cache' && backend !== 'indexeddb' && backend !== 'opfs') {
    throw new Error('Choose Cache API, IndexedDB, or OPFS.')
  }
  return { ...prebuiltAppConfig, cacheBackend: backend }
}
