import './types.js'

let loaded: Awaited<ReturnType<typeof import('../engine/load.js').openEngine>> | undefined
const status = document.querySelector('[role="status"]')!

window.harness = {
  async open(modelId, contextWindow = 8192) {
    if (loaded) throw new Error('A model is already loaded.')
    const [{ openEngine, cacheConfig }, { createBrowserBridge }] = await Promise.all([
      import('../engine/load.js'),
      import('strands-web-llm'),
    ])
    loaded = await openEngine(
      'worker',
      modelId,
      contextWindow,
      await cacheConfig('cache'),
      (report) => {
        status.textContent = report.text
      },
    )
    window.webllmBridge = createBrowserBridge(loaded.engine)
    status.textContent = 'Model loaded'
  },
  async close() {
    try {
      await window.webllmBridge?.close()
    } finally {
      try {
        await loaded?.close()
      } finally {
        loaded = undefined
      }
    }
  },
}
