import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMProviderOptions } from 'pi-ai-web-llm'
import { chatOptions } from '../../chat-options.js'
import { sampling } from '../../sampling.js'
import runtimeUrl from 'pi-ai-web-llm/runtime/gemma-4-e2b.wasm?url'
import runtime from 'pi-ai-web-llm/runtime/manifest.json'

const contextWindow = 32768
let engine: MLCEngineInterface | undefined
let worker: Worker | undefined
let modelId = ''

export async function load(id: string, useWorker = true, onProgress?: (text: string) => void) {
  const { MLCEngine, CreateWebWorkerMLCEngine, modelVersion, prebuiltAppConfig } =
    await import('@mlc-ai/web-llm')
  await unload()
  modelId = id
  const runtimeVersion = `v${runtime.webllm.replaceAll('.', '_')}/base`
  const appConfig =
    modelVersion === runtimeVersion
      ? {
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
        }
      : prebuiltAppConfig
  const options = {
    appConfig,
    initProgressCallback: (progress: import('@mlc-ai/web-llm').InitProgressReport) => {
      onProgress?.(progress.text)
    },
  }
  try {
    if (useWorker) {
      worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      engine = await CreateWebWorkerMLCEngine(worker, id, options, {
        ...chatOptions(id),
        context_window_size: contextWindow,
      })
    } else {
      engine = new MLCEngine(options)
      await engine.reload(id, { ...chatOptions(id), context_window_size: contextWindow })
    }
  } catch (error) {
    await unload().catch((cleanupError: unknown) => console.error(cleanupError))
    throw error
  }
}

export async function provider(
  enabled: boolean,
  options: Pick<WebLLMProviderOptions, 'thinkingPrefilled' | 'request'> = {},
) {
  if (!engine) throw new Error('Load a model first.')
  const [{ createModels }, { createWebLLMProvider }] = await Promise.all([
    import('@earendil-works/pi-ai/models'),
    import('pi-ai-web-llm'),
  ])
  const models = createModels()
  const source = createWebLLMProvider({
    engine,
    modelId,
    thinkingPrefilled: options.thinkingPrefilled ?? modelId.startsWith('Qwen3.5-'),
    contextWindow,
    request: { ...sampling(modelId, enabled), max_tokens: 4096, seed: 7, ...options.request },
  })
  models.setProvider(source)
  return { models, model: source.getModels()[0]!, engine }
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
