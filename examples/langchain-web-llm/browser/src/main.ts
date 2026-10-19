import { sampling } from '../../../shared/sampling.js'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
type LangChainAdapter = typeof import('langchain-web-llm')
import runtimeUrl from 'langchain-web-llm/runtime/gemma-4-e2b.wasm?url'
import runtime from 'langchain-web-llm/runtime/manifest.json'

const status = document.querySelector<HTMLParagraphElement>('#status')!
const answer = document.querySelector<HTMLPreElement>('#answer')!
const details = document.querySelector<HTMLPreElement>('#details')!
const modelSelect = document.querySelector<HTMLSelectElement>('#model')!
const thinking = document.querySelector<HTMLInputElement>('#thinking')!
let engine: MLCEngineInterface | undefined
let modelId = ''
let contextWindowSize: number | undefined
let cancel: AbortController | undefined
let worker: Worker | undefined
let adapter: LangChainAdapter | undefined

async function load(id: string, useWorker = true) {
  const [{ CreateMLCEngine, CreateWebWorkerMLCEngine, prebuiltAppConfig }, langChainAdapter] =
    await Promise.all([import('@mlc-ai/web-llm'), import('langchain-web-llm')])
  if (engine) {
    const previousBridge = window.webllmLangChainBridge
    const previousEngine = engine
    const previousWorker = worker
    engine = undefined
    worker = undefined
    delete window.webllmLangChainBridge
    try {
      await previousBridge?.close()
    } finally {
      try {
        await previousEngine.unload()
      } finally {
        previousWorker?.terminate()
      }
    }
  }
  modelId = id
  const appConfig = {
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
  contextWindowSize = appConfig.model_list.find((model) => model.model_id === id)?.overrides
    ?.context_window_size
  const options = {
    appConfig,
    initProgressCallback: (progress: import('@mlc-ai/web-llm').InitProgressReport) => {
      status.textContent = progress.text
    },
  }
  engine = undefined
  try {
    if (useWorker) {
      worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      engine = await CreateWebWorkerMLCEngine(worker, id, options)
    } else engine = await CreateMLCEngine(id, options)
  } catch (error) {
    worker?.terminate()
    worker = undefined
    throw error
  }
  adapter = langChainAdapter
  window.webllmLangChainBridge = adapter.createBrowserBridge(engine)
  thinking.disabled = !model().profile.reasoningOutput
  thinking.checked = !thinking.disabled
  status.textContent = 'Ready'
}

function model(enabled = thinking.checked) {
  if (!engine) throw new Error('Load a model first.')
  if (!adapter) throw new Error('Load a model first.')
  return new adapter.ChatWebLLM({
    engine,
    modelId,
    ...(contextWindowSize !== undefined && { profile: { maxInputTokens: contextWindowSize } }),
    request: { ...sampling(modelId, enabled), max_tokens: 4096, seed: 7 },
  })
}

async function run(message: string, enabled = thinking.checked, signal?: AbortSignal) {
  const [{ createAgent }, { tool }, { z }] = await Promise.all([
    import('langchain'),
    import('@langchain/core/tools'),
    import('zod'),
  ])
  const lookup = tool(({ code }) => `The answer for ${code} is 42.`, {
    name: 'lookup',
    description: 'Look up the answer for a code.',
    schema: z.object({ code: z.string() }),
  })
  const agent = createAgent({ model: model(enabled), tools: [lookup] })
  return agent.invoke(
    { messages: [{ role: 'user', content: message }] },
    { ...(signal && { signal }) },
  )
}

document.querySelector<HTMLButtonElement>('#load')!.onclick = () => {
  busy(true)
  void load(modelSelect.value, document.querySelector<HTMLInputElement>('#worker')!.checked)
    .catch((error) => {
      status.textContent = String(error)
    })
    .finally(() => busy(false))
}
document.querySelector<HTMLButtonElement>('#cancel')!.onclick = () => cancel?.abort()
document.querySelector<HTMLFormElement>('#chat')!.onsubmit = (event) => {
  event.preventDefault()
  busy(true)
  cancel = new AbortController()
  document.querySelector<HTMLButtonElement>('#cancel')!.disabled = false
  answer.textContent = ''
  details.textContent = ''
  status.textContent = 'Running'
  const message = document.querySelector<HTMLInputElement>('#message')!.value
  void run(message, thinking.checked, cancel.signal)
    .then(async (result) => {
      const { AIMessage } = await import('@langchain/core/messages')
      answer.textContent = result.messages.at(-1)?.text ?? ''
      details.textContent = result.messages
        .filter(AIMessage.isInstance)
        .map((message) =>
          JSON.stringify({ content: message.content, tools: message.tool_calls }, null, 2),
        )
        .join('\n')
      status.textContent = 'Done'
    })
    .catch((error) => {
      status.textContent = String(error)
    })
    .finally(() => busy(false))
}

// Browser checks exercise the same engine and provider used by this example.
export const checks = {
  contextWindowSize: () => contextWindowSize,
  engine: () => {
    if (!engine) throw new Error('Load a model first.')
    return engine
  },
  load,
  run,
  model,
  embeddings: () => {
    if (!engine) throw new Error('Load an embedding model first.')
    if (!adapter) throw new Error('Load an embedding model first.')
    return new adapter.WebLLMEmbeddings({ engine, modelId })
  },
}
Object.assign(window, { checks })

declare global {
  interface Window {
    checks: typeof checks
  }
}

function busy(value: boolean) {
  if (!value) document.querySelector<HTMLButtonElement>('#cancel')!.disabled = true
  for (const button of document.querySelectorAll<HTMLButtonElement>('button'))
    if (button.id !== 'cancel') button.disabled = value
  modelSelect.disabled = value
}
