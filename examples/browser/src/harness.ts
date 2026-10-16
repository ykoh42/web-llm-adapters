import { Message } from '@strands-agents/sdk'
import type { HarnessRequest, HarnessResult, HarnessCountRequest } from './harness-api.js'
import { WebLLMModel } from 'strands-webllm'
import { openEngine, cacheConfig } from './engines.js'

let loaded: Awaited<ReturnType<typeof openEngine>> | undefined
interface Call {
  stream: ReturnType<WebLLMModel['stream']>
  controller: AbortController
  closing?: Promise<void>
}
const calls = new Map<string, Call>()
const status = document.querySelector('[role="status"]')!

const harness = {
  async open(modelId: string | string[], contextWindow = 8192) {
    if (loaded) throw new Error('A model is already loaded.')
    loaded = await openEngine('worker', modelId, contextWindow, cacheConfig('cache'), (report) => {
      status.textContent = report.text
    })
    status.textContent = 'Model loaded'
  },
  async count({ messages, config, options }: HarnessCountRequest) {
    if (!loaded) throw new Error('Load a model first.')
    const model = new WebLLMModel({ engine: loaded.engine, ...config })
    return model.countTokens(messages.map(Message.fromJSON), options)
  },
  async start({ messages, config, options }: HarnessRequest) {
    if (!loaded) throw new Error('Load a model first.')
    const id = crypto.randomUUID()
    const controller = new AbortController()
    const model = new WebLLMModel({ engine: loaded.engine, ...config })
    const stream = model.stream(messages.map(Message.fromJSON), {
      ...options,
      cancelSignal: controller.signal,
    })
    calls.set(id, { stream, controller })
    return id
  },
  async next(id: string): Promise<HarnessResult> {
    const call = calls.get(id)
    try {
      return call ? await call.stream.next() : { done: true, value: undefined }
    } catch (error) {
      return {
        error: {
          name: error instanceof Error ? error.name : 'Error',
          message: error instanceof Error ? error.message : String(error),
        },
      }
    }
  },
  stop(id: string): Promise<void> {
    const call = calls.get(id)
    if (!call) return Promise.resolve()
    if (call.closing) return call.closing
    call.controller.abort()
    call.closing = (async () => {
      try {
        await call.stream.return(undefined)
      } finally {
        calls.delete(id)
      }
    })()
    return call.closing
  },
  async close() {
    try {
      await Promise.all([...calls.keys()].map((id) => harness.stop(id)))
    } finally {
      await loaded?.close()
      loaded = undefined
    }
  },
}

window.harness = harness
