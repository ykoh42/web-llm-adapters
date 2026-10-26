import { type AppConfig, type InitProgressCallback, type MLCEngineInterface } from '@mlc-ai/web-llm'
import { chatOptions } from '../../../chat-options.js'
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
  const { MLCEngine, WebWorkerMLCEngine, ServiceWorkerMLCEngine, ExtensionServiceWorkerMLCEngine } =
    await import('@mlc-ai/web-llm')
  const config = { appConfig, initProgressCallback }
  let worker: Worker | undefined
  let engine: MLCEngineInterface | undefined
  let closing: Promise<void> | undefined
  const close = () =>
    (closing ??= (async () => {
      try {
        await engine?.unload()
      } finally {
        worker?.terminate()
        if (engine instanceof ExtensionServiceWorkerMLCEngine) engine.port.disconnect()
      }
    })())
  try {
    switch (kind) {
      case 'worker':
        worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
        engine = new WebWorkerMLCEngine(worker, config)
        break
      case 'service-worker':
        await navigator.serviceWorker.register(serviceWorkerUrl, { type: 'module', scope: '/' })
        await waitForController(new URL(serviceWorkerUrl, location.href).href)
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
    if (!engine) throw new Error('Could not create the WebLLM engine.')
    const optionsFor = (id: string) => ({
      ...chatOptions(id),
      ...(contextWindow !== undefined && { context_window_size: contextWindow }),
    })
    const options = Array.isArray(modelId) ? modelId.map(optionsFor) : optionsFor(modelId)
    await engine.reload(modelId, options)
    return { engine, close }
  } catch (error) {
    await close().catch((cleanupError: unknown) => console.error(cleanupError))
    throw error
  }
}

function waitForController(scriptURL: string): Promise<void> {
  const controlsPage = () => navigator.serviceWorker.controller?.scriptURL === scriptURL
  if (controlsPage()) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const ready = () => {
      if (!controlsPage()) return
      clearTimeout(timer)
      navigator.serviceWorker.removeEventListener('controllerchange', ready)
      resolve()
    }
    const timer = setTimeout(() => {
      navigator.serviceWorker.removeEventListener('controllerchange', ready)
      reject(new Error('Service worker did not take control. Reload the page and try again.'))
    }, 15_000)
    navigator.serviceWorker.addEventListener('controllerchange', ready)
    if (controlsPage()) ready()
  })
}

export async function cacheConfig(backend: string): Promise<AppConfig> {
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
  const { prebuiltAppConfig } = await import('@mlc-ai/web-llm')
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
