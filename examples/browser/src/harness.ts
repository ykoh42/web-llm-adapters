import { Message } from '@strands-agents/sdk'
import type { HarnessRequest } from './harness-api.js'
import { WebLLMModel } from 'strands-webllm'
import { openEngine, cacheConfig } from './engines.js'

let loaded: Awaited<ReturnType<typeof openEngine>> | undefined
let stream: ReturnType<WebLLMModel['stream']> | undefined
let controller: AbortController | undefined
let closing: Promise<void> | undefined
const status = document.querySelector('[role="status"]')!

const harness = {
  async open(modelId: string) {
    if (loaded) throw new Error('A model is already loaded.')
    loaded = await openEngine('worker', modelId, 8192, cacheConfig('cache'), (report) => {
      status.textContent = report.text
    })
    status.textContent = 'Model loaded'
  },
  async start({ messages, config, options }: HarnessRequest) {
    await closing
    if (!loaded) throw new Error('Load a model first.')
    if (stream) throw new Error('A stream is already active.')
    controller = new AbortController()
    const model = new WebLLMModel({ engine: loaded.engine, ...config })
    stream = model.stream(messages.map(Message.fromJSON), {
      ...options,
      cancelSignal: controller.signal,
    })
  },
  async next() {
    if (!stream) throw new Error('No active stream.')
    return stream.next()
  },
  stop(): Promise<void> {
    if (closing) return closing
    if (!stream) return Promise.resolve()
    const current = stream
    controller?.abort()
    closing = (async () => {
      try {
        await current.return(undefined)
      } finally {
        stream = undefined
        controller = undefined
        closing = undefined
      }
    })()
    return closing
  },
  async close() {
    try {
      await harness.stop()
    } finally {
      await loaded?.close()
      loaded = undefined
    }
  },
}

window.harness = harness
