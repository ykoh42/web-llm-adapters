import { MLCEngine, WebWorkerMLCEngine, type MLCEngineInterface } from '@mlc-ai/web-llm'
import { withEngine } from 'strands-web-llm'
import { qwen } from './catalog.js'
import { FixedToken } from './fixed-token.js'

async function forward(engine: MLCEngineInterface) {
  return withEngine(engine, async (engine) => {
    const tokens = [await engine.forwardTokensAndSample([1], true, qwen)]
    for (let i = 0; i < 2; i++) {
      tokens.push(await engine.forwardTokensAndSample([tokens.at(-1)!], false, qwen))
    }
    return tokens
  })
}

export async function processor() {
  const hook = new FixedToken()
  const engine = new MLCEngine({
    logLevel: 'SILENT',
    logitProcessorRegistry: new Map([[qwen, hook]]),
  })
  let main
  try {
    await engine.reload(qwen, { context_window_size: 1024 })
    const tokens = await forward(engine)
    const sampled = [...hook.sampled]
    const resets = hook.resets
    const stats = await engine.runtimeStatsText(qwen)
    await engine.resetChat(false, qwen)
    main = {
      tokens,
      sampled,
      resetCalled: hook.resets > resets,
      cleared: hook.sampled.length === 0,
      stats,
    }
  } finally {
    await engine.unload()
  }

  const worker = new Worker(new URL('./processor-worker.ts', import.meta.url), { type: 'module' })
  const client = new WebWorkerMLCEngine(worker, { logLevel: 'SILENT' })
  try {
    await client.reload(qwen, { context_window_size: 1024 })
    return { main, worker: { tokens: await forward(client) } }
  } finally {
    try {
      await client.unload()
    } finally {
      worker.terminate()
    }
  }
}
