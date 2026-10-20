import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { sampling } from '../../sampling.js'
import runtimeUrl from 'pi-ai-web-llm/runtime/gemma-4-e2b.wasm?url'
import runtime from 'pi-ai-web-llm/runtime/manifest.json'

const contextWindow = 32768
let engine: MLCEngineInterface | undefined
let worker: Worker | undefined
let modelId = ''

export async function load(id: string, useWorker = true, onProgress?: (text: string) => void) {
  const { CreateMLCEngine, CreateWebWorkerMLCEngine, prebuiltAppConfig } =
    await import('@mlc-ai/web-llm')
  await unload()
  modelId = id
  const options = {
    appConfig: {
      ...prebuiltAppConfig,
      model_list: prebuiltAppConfig.model_list.map((model) =>
        model.model_id === runtime.model_id
          ? {
              ...model,
              model_lib: new URL(runtimeUrl, location.href).href,
              integrity: { ...model.integrity, model_lib: runtime.integrity },
            }
          : model,
      ),
    },
    initProgressCallback: (progress: import('@mlc-ai/web-llm').InitProgressReport) => {
      onProgress?.(progress.text)
    },
  }
  try {
    if (useWorker) {
      worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      engine = await CreateWebWorkerMLCEngine(worker, id, options, {
        context_window_size: contextWindow,
      })
    } else engine = await CreateMLCEngine(id, options, { context_window_size: contextWindow })
  } catch (error) {
    worker?.terminate()
    worker = undefined
    throw error
  }
}

export async function provider(enabled: boolean) {
  if (!engine) throw new Error('Load a model first.')
  const [{ createModels }, { createWebLLMProvider }] = await Promise.all([
    import('@earendil-works/pi-ai/models'),
    import('pi-ai-web-llm'),
  ])
  const models = createModels()
  const source = createWebLLMProvider({
    engine,
    modelId,
    contextWindow,
    request: { ...sampling(modelId, enabled), max_tokens: 4096, seed: 7 },
  })
  models.setProvider(source)
  return { models, model: source.getModels()[0]! }
}

export async function unload() {
  try {
    await engine?.unload()
  } finally {
    worker?.terminate()
    engine = undefined
    worker = undefined
  }
}
