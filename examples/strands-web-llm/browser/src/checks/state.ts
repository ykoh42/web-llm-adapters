import { MLCEngine } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel, withEngine } from 'strands-web-llm'
import { qwen, record } from './catalog.js'

export async function state() {
  const progress: number[] = []
  const callback = (report: { progress: number }) => {
    progress.push(report.progress)
  }
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  try {
    await withEngine(engine, async (engine) => {
      engine.setInitProgressCallback(callback)
      engine.setAppConfig({ model_list: [record(qwen)] })
      engine.setLogLevel('WARN')
      await engine.reload(qwen, { context_window_size: 2048 })
    })
    const model = new WebLLMModel({
      engine,
      modelId: qwen,
      contextWindowLimit: 2048,
      maxTokens: 64,
      temperature: 0,
      request: { extra_body: { enable_thinking: false } },
    })
    const agent = new Agent({
      model,
      printer: false,
      systemPrompt: 'Remember the code word the user gives you.',
    })
    await agent.invoke('The code word is pineapple. Reply OK.')
    const result = await agent.invoke('What is the code word?')
    const text = result.lastMessage.content
      .filter((block) => block.type === 'textBlock')
      .map((block) => block.text)
      .join('\n')
    return await withEngine(engine, async (engine) => {
      const message = await engine.getMessage(qwen)
      const gpu = await engine.getGPUVendor()
      const maxBuffer = await engine.getMaxStorageBufferBindingSize()
      const stats = await engine.runtimeStatsText(qwen)
      await engine.resetChat(false, qwen)
      return {
        progress,
        callbackMatches: engine.getInitProgressCallback() === callback,
        text,
        message,
        afterReset: await engine.getMessage(qwen),
        gpu,
        maxBuffer,
        stats,
        messages: agent.messages.length,
      }
    })
  } finally {
    await withEngine(engine, (engine) => engine.unload())
  }
}
