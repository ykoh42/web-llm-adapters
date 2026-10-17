import './harness-api.js'
import { createBrowserBridge } from 'strands-web-llm'
import { openEngine, cacheConfig } from './engines.js'

let loaded: Awaited<ReturnType<typeof openEngine>> | undefined
const status = document.querySelector('[role="status"]')!

window.harness = {
  async open(modelId, contextWindow = 8192) {
    if (loaded) throw new Error('A model is already loaded.')
    loaded = await openEngine('worker', modelId, contextWindow, cacheConfig('cache'), (report) => {
      status.textContent = report.text
    })
    window.webllmBridge = createBrowserBridge(loaded.engine)
    status.textContent = 'Model loaded'
  },
  async close() {
    try {
      await window.webllmBridge?.close()
    } finally {
      await loaded?.close()
      loaded = undefined
    }
  },
}
