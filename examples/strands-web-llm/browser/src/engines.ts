import {
  MLCEngine,
  WebWorkerMLCEngine,
  ServiceWorkerMLCEngine,
  ExtensionServiceWorkerMLCEngine,
  prebuiltAppConfig,
} from '@mlc-ai/web-llm'
import type { AppConfig, InitProgressCallback, MLCEngineInterface } from '@mlc-ai/web-llm'
import serviceWorkerUrl from './service-worker.ts?worker&url'
import { runtimeConfig } from './runtime.js'

export async function openEngine(
  kind: string,
  modelId: string | string[],
  contextWindow: number | undefined,
  appConfig: AppConfig,
  initProgressCallback: InitProgressCallback,
) {
  if (appConfig.cacheBackend === 'cross-origin' && kind !== 'main') {
    throw new Error('Use Main thread for the Cross-Origin Storage extension.')
  }
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
    case 'extension':
      if (!globalThis.chrome?.runtime?.id)
        throw new Error('Open the example inside its Chrome extension.')
      engine = new ExtensionServiceWorkerMLCEngine(config)
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
      if (engine instanceof ExtensionServiceWorkerMLCEngine) engine.port.disconnect()
    }
  }
  try {
    const chatOptions = contextWindow ? { context_window_size: contextWindow } : undefined
    const options =
      Array.isArray(modelId) && chatOptions ? modelId.map(() => chatOptions) : chatOptions
    await engine.reload(modelId, options)
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
  if (
    backend !== 'cache' &&
    backend !== 'indexeddb' &&
    backend !== 'opfs' &&
    backend !== 'cross-origin'
  ) {
    throw new Error('Choose Cache API, IndexedDB, or OPFS.')
  }
  if (backend === 'cross-origin' && !hasSharedCache()) {
    throw new Error('Install the Cross-Origin Storage extension and open this page on localhost.')
  }
  return runtimeConfig({ ...prebuiltAppConfig, cacheBackend: backend })
}

// WebLLM otherwise falls back silently to Cache API. Show the missing requirement instead.
function hasSharedCache() {
  return (
    'crossOriginStorage' in navigator &&
    typeof navigator.crossOriginStorage === 'object' &&
    navigator.crossOriginStorage !== null &&
    'getFileHandle' in navigator.crossOriginStorage &&
    typeof navigator.crossOriginStorage.getFileHandle === 'function'
  )
}
