import { sampling } from '../../../shared/sampling.js'
import {
  CreateMLCEngine,
  CreateWebWorkerMLCEngine,
  prebuiltAppConfig,
  type MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { AIMessage } from '@langchain/core/messages'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { ChatWebLLM, WebLLMEmbeddings, createBrowserBridge } from 'langchain-web-llm'
import runtimeUrl from 'langchain-web-llm/runtime/gemma-4-e2b.wasm?url'
import runtime from 'langchain-web-llm/runtime/manifest.json'

const status = document.querySelector<HTMLParagraphElement>('#status')!
const answer = document.querySelector<HTMLPreElement>('#answer')!
const details = document.querySelector<HTMLPreElement>('#details')!
const modelSelect = document.querySelector<HTMLSelectElement>('#model')!
const thinking = document.querySelector<HTMLInputElement>('#thinking')!
let engine: MLCEngineInterface | undefined
let modelId = ''
let cancel: AbortController | undefined
let worker: Worker | undefined

async function load(id: string, useWorker = false) {
  if (engine) {
    await window.webllmLangChainBridge.close()
    await engine.unload()
    worker?.terminate()
    worker = undefined
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
  const options = {
    appConfig,
    initProgressCallback: (progress: import('@mlc-ai/web-llm').InitProgressReport) => {
      status.textContent = progress.text
    },
  }
  const context = id.startsWith('snowflake-arctic-embed-')
    ? undefined
    : { context_window_size: 32768 }
  engine = undefined
  try {
    if (useWorker) {
      worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      engine = await CreateWebWorkerMLCEngine(worker, id, options, context)
    } else engine = await CreateMLCEngine(id, options, context)
  } catch (error) {
    worker?.terminate()
    worker = undefined
    throw error
  }
  window.webllmLangChainBridge = createBrowserBridge(engine)
  thinking.disabled = !model().profile.reasoningOutput
  thinking.checked = !thinking.disabled
  status.textContent = 'Ready'
}

function model(enabled = thinking.checked) {
  if (!engine) throw new Error('Load a model first.')
  return new ChatWebLLM({
    engine,
    modelId,
    profile: { maxInputTokens: 32768 },
    request: { ...sampling(modelId, enabled), max_tokens: 4096, seed: 7 },
  })
}

async function run(message: string, enabled = thinking.checked, signal?: AbortSignal) {
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
    .then((result) => {
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
  engine: () => {
    if (!engine) throw new Error('Load a model first.')
    return engine
  },
  load,
  run,
  model,
  embeddings: () => {
    if (!engine) throw new Error('Load an embedding model first.')
    return new WebLLMEmbeddings({ engine, modelId })
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
