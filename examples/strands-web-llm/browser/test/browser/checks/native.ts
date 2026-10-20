import { MLCEngine, prebuiltAppConfig } from '@mlc-ai/web-llm'
import { runtimeConfig } from '../../../src/engine/runtime.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import type { ExampleInput } from '../../../src/examples/output.js'

async function embeddings(input: ExampleInput) {
  const result = await input.engine.embeddings.create({
    model: input.modelId,
    input: input.prompt,
  })
  const vector = result.data[0]!.embedding
  input.write(
    JSON.stringify({ dimensions: vector.length, firstValues: vector.slice(0, 8) }, null, 2),
  )
  input.info(result.usage)
}

async function completions(input: ExampleInput) {
  // This API uses a raw prompt without a chat template. It also allows multiple choices.
  const result = await input.engine.completions.create({
    model: input.modelId,
    prompt: input.prompt,
    max_tokens: 64,
    temperature: 0.7,
    seed: 42,
    n: 2,
  })
  input.write(result.choices.map((choice) => choice.text).join('\n\n---\n\n'))
  input.info(result.usage)
}

async function resume(input: ExampleInput) {
  const sessionId = crypto.randomUUID()
  let parts = 0
  const chunks = await input.engine.chat.completions.create({
    model: input.modelId,
    messages: [{ role: 'user', content: input.prompt }],
    max_tokens: 96,
    temperature: 0,
    stream: true,
    extra_body: { resumable: { enabled: true, sessionId, checkpointIntervalTokens: 32 } },
  })
  for await (const chunk of chunks) {
    const text = chunk.choices[0]?.delta.content
    if (text) {
      input.write(text)
      if (++parts === 8) input.engine.interruptGenerate()
    }
  }
  if (input.signal.aborted) return
  input.info(await input.engine.listResumableSessions())
  const saved = await input.engine.resumeChatCompletion(sessionId)
  if ('recoveredText' in saved) input.info(saved)
  input.signal.throwIfAborted()
  const continued = await input.engine.resumeChatCompletion(sessionId, {
    continueGeneration: true,
    stream: true,
  })
  if ('recoveredText' in continued) throw new Error('Expected a resumed stream.')
  await showChunks(continued, input)
  if (!input.signal.aborted) await input.engine.deleteResumableSession(sessionId)
}

async function showChunks(chunks: AsyncIterable<ChatCompletionChunk>, input: ExampleInput) {
  for await (const chunk of chunks) {
    const text = chunk.choices[0]?.delta.content
    if (text) input.write(text)
    if (chunk.usage) input.info(chunk.usage)
  }
}

const cases = {
  embeddings: {
    modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
    prompt: 'A small browser agent.',
    contextWindow: 512,
    run: embeddings,
  },
  completions: {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    prompt: 'The sky is',
    contextWindow: 2048,
    run: completions,
  },
  resume: {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    prompt: 'Count from 1 to 30, with spaces between numbers. Do not explain.',
    contextWindow: 2048,
    run: resume,
  },
}

export async function native(name: keyof typeof cases) {
  const check = cases[name]
  const engine = new MLCEngine({ appConfig: runtimeConfig(prebuiltAppConfig) })
  const reports: unknown[] = []
  let text = ''
  try {
    await engine.reload(check.modelId, { context_window_size: check.contextWindow })
    await check.run({
      engine,
      modelId: check.modelId,
      prompt: check.prompt,
      file: undefined,
      signal: new AbortController().signal,
      write: (part) => {
        text += part
      },
      info: (value) => reports.push(value),
    })
    return { text, reports }
  } finally {
    await engine.unload()
  }
}
