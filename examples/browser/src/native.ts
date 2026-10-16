import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import type { ExampleInput } from './output.js'

export async function embeddings(input: ExampleInput) {
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

export async function completions(input: ExampleInput) {
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

export async function rawAudio(input: ExampleInput) {
  if (!input.file) throw new Error('Choose an audio file first.')
  const audio = new AudioContext({ sampleRate: 16000 })
  try {
    const decoded = await audio.decodeAudioData(await input.file.arrayBuffer())
    input.signal.throwIfAborted()
    const samples = new Float32Array(decoded.length)
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const values = decoded.getChannelData(channel)
      for (let i = 0; i < samples.length; i++) samples[i]! += values[i]! / decoded.numberOfChannels
    }
    const chunks = await input.engine.chat.completions.create({
      model: input.modelId,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: input.prompt },
            {
              type: 'input_audio',
              input_audio: {
                format: 'pcm_f32',
                data: samples,
                sample_rate: decoded.sampleRate,
              },
            },
          ],
        },
      ],
      max_tokens: 256,
      stream: true,
    })
    await showChunks(chunks, input)
  } finally {
    await audio.close()
  }
}

export async function resume(input: ExampleInput) {
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
