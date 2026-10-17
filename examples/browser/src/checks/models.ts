import { MLCEngine } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel, withEngine } from 'strands-webllm'
import { qwen, record } from './catalog.js'

export async function models() {
  const first = { ...record(qwen), model_id: 'Qwen3-first' }
  const second = { ...record(qwen), model_id: 'Qwen3-second' }
  const embed = record('snowflake-arctic-embed-s-q0f32-MLC-b4')
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  engine.setAppConfig({ model_list: [first, second, embed] })
  try {
    await engine.reload(
      [first.model_id, second.model_id, embed.model_id],
      [{ context_window_size: 1024 }, { context_window_size: 1024 }, {}],
    )
    const seen = new Set<string>()
    const model = new WebLLMModel({
      engine,
      modelId: first.model_id,
      contextWindowLimit: 1024,
      maxTokens: 32,
      request: { extra_body: { enable_thinking: false } },
      onChunk: (chunk) => seen.add(chunk.model),
    })
    await new Agent({ model, printer: false }).invoke('Say hello.')
    model.updateConfig({ modelId: second.model_id })
    await new Agent({ model, printer: false }).invoke('Say hello.')
    const { replies, vector } = await withEngine(engine, async (engine) => {
      const replies = await Promise.all(
        [first, second].map((item) =>
          engine.chat.completions.create({
            model: item.model_id,
            messages: [{ role: 'user', content: 'Say hello.' }],
            max_tokens: 32,
            extra_body: { enable_thinking: false },
          }),
        ),
      )
      const vector = await engine.embeddings.create({ model: embed.model_id, input: 'Hello' })
      return { replies, vector }
    })
    return {
      selected: [...seen],
      parallel: replies.map((reply) => ({
        model: reply.model,
        text: reply.choices[0]?.message.content,
      })),
      dimensions: vector.data[0]?.embedding.length,
    }
  } finally {
    await engine.unload()
  }
}
