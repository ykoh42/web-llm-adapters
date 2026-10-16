import { IntegrityError, MLCEngine, type ModelIntegrity } from '@mlc-ai/web-llm'
import { record } from './catalog.js'

async function checksum(url: string) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Cannot read artifact: ${url}`)
  const hash = await crypto.subtle.digest('SHA-256', await response.arrayBuffer())
  return 'sha256-' + btoa(String.fromCharCode(...new Uint8Array(hash)))
}

export async function integrity() {
  const model = { ...record('gemma-4-E2B-it-q4f16_1-MLC'), model_id: 'custom-gemma' }
  const base = model.model + '/resolve/main/'
  // For a real release, pin hashes from a trusted publisher instead of computing them here.
  const valid: ModelIntegrity = {
    config: await checksum(base + 'mlc-chat-config.json'),
    model_lib: await checksum(model.model_lib),
    tokenizer: { 'tokenizer.json': await checksum(base + 'tokenizer.json') },
  }
  const bad = 'sha256-' + btoa(String.fromCharCode(...new Uint8Array(32)))
  const failures = []
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  try {
    engine.setAppConfig({ model_list: [{ ...model, integrity: valid }] })
    await engine.reload(model.model_id, { context_window_size: 2048 })
    const result = await engine.chat.completions.create({
      model: model.model_id,
      messages: [{ role: 'user', content: 'Say hello.' }],
      max_tokens: 32,
    })
    for (const field of ['config', 'model_lib', 'tokenizer'] as const) {
      const broken = { ...valid }
      if (field === 'tokenizer') broken.tokenizer = { 'tokenizer.json': bad }
      else broken[field] = bad
      engine.setAppConfig({ model_list: [{ ...model, integrity: broken }] })
      try {
        await engine.reload(model.model_id, { context_window_size: 2048 })
        throw new Error(`Invalid ${field} checksum was accepted.`)
      } catch (error) {
        if (!(error instanceof IntegrityError)) throw error
        failures.push({ field, url: error.url, expected: error.expected, actual: error.actual })
      }
    }
    return {
      manifest: model.model_manifest,
      text: result.choices[0]?.message.content,
      checksums: valid,
      failures,
    }
  } finally {
    await engine.unload()
  }
}
