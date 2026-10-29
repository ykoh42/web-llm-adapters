import { modelCheck } from '../../../../../scripts/release/support.js'
import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { scrubUnsupportedMultimodalContent } from 'deepagents'
import { z } from 'zod'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'

test(
  'PCM and WAV audio reach WebLLM through local and browser LangChain models',
  modelCheck('gemma-4-E2B-it-q4f16_1-MLC', 'audio'),
  async ({ page }) => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    await page.evaluate((id) => window.checks.load(id), modelId)
    const prompt = 'Describe any sound you hear in one short sentence.'
    const request = { temperature: 0, top_p: 1, repetition_penalty: 1, max_tokens: 4096, seed: 7 }
    const samples = Float32Array.from(
      { length: 16000 },
      (_, index) => 0.1 * Math.sin((2 * Math.PI * 440 * index) / 16000),
    )
    const local = await page.evaluate(
      async ({ prompt, samples, request }) => {
        const engine = window.checks.engine()
        window.audioRequests = []
        const endpoint = engine.chat.completions
        endpoint.create = new Proxy(endpoint.create, {
          apply(target, self, args) {
            window.audioRequests.push(structuredClone(args[0]))
            return Reflect.apply(target, self, args)
          },
        })
        const messages = [
          {
            role: 'user' as const,
            content: [
              { type: 'text' as const, text: prompt },
              {
                type: 'input_audio' as const,
                input_audio: {
                  format: 'pcm_f32' as const,
                  data: new Float32Array(samples),
                  sample_rate: 16000,
                },
              },
            ],
          },
        ]
        await engine.resetChat()
        const result = await window.checks.model(false).invoke(messages, { request })
        return {
          text: result.text,
          usage: result.usage_metadata,
          native: await engine.getMessage(),
        }
      },
      { prompt, samples: [...samples], request },
    )

    await page.evaluate(() => window.checks.engine().resetChat())
    let native = ''
    let usage: { prompt_tokens: number; completion_tokens: number } | undefined
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request,
      onChunk: (chunk) => {
        const choice = chunk.choices[0]
        if (choice && 'delta' in choice) native += choice.delta.content ?? ''
        if (chunk.usage) usage = chunk.usage
      },
    })
    const remote = await model.invoke([
      new HumanMessage({
        content: [
          { type: 'text', text: prompt },
          {
            type: 'input_audio',
            input_audio: { format: 'pcm_f32', data: samples, sample_rate: 16000 },
          },
        ],
      }),
    ])

    expect(local.text).toBe(local.native)
    expect(local.text.length).toBeGreaterThan(0)
    expect(local.usage?.input_tokens).toBeGreaterThan(0)
    expect(remote.text).toBe(native)
    expect(usage?.prompt_tokens).toBeGreaterThan(0)
    expect(remote.usage_metadata?.input_tokens).toBe(usage?.prompt_tokens)
    expect(remote.usage_metadata?.output_tokens).toBe(usage?.completion_tokens)
    const wav = await page.evaluate(() => {
      const content = window.audioRequests[1]?.messages[0]?.content
      if (!Array.isArray(content)) throw new Error('Expected audio content.')
      const audio = content.find((part) => part.type === 'input_audio')
      if (audio?.type !== 'input_audio' || audio.input_audio.format !== 'wav')
        throw new Error('Expected WAV audio.')
      return audio.input_audio.data
    })
    const url = 'DATA:audio/wav;base64,' + wav
    const message = new HumanMessage({
      content: [
        { type: 'text', text: prompt },
        { type: 'audio', url },
      ],
    })
    await page.evaluate(() => window.checks.engine().resetChat())
    await model.invoke([message], { request: { ...request, max_tokens: 1 } })
    expect(message.content).toEqual([
      { type: 'text', text: prompt },
      { type: 'audio', url },
    ])
    const inputs = await page.evaluate(() =>
      window.audioRequests.map((request) => {
        const content = request.messages[0]?.content
        if (!Array.isArray(content)) throw new Error('Expected audio content.')
        const part = content.find((part) => part.type === 'input_audio')
        if (part?.type !== 'input_audio') throw new Error('Expected an audio input.')
        if (part.input_audio.format === 'wav') {
          const encoded = part.input_audio.data.replace(
            /^data:audio\/(?:wav|wave|x-wav);base64,/i,
            '',
          )
          const bytes = Uint8Array.from(atob(encoded), (value) => value.charCodeAt(0))
          const view = new DataView(bytes.buffer)
          if (view.getUint16(20, true) !== 3 || view.getUint16(22, true) !== 1)
            throw new Error('Expected mono Float32 WAV.')
          return {
            format: 'wav',
            sampleRate: view.getUint32(24, true),
            samples: Array.from({ length: view.getUint32(52, true) / 4 }, (_, index) =>
              view.getFloat32(56 + index * 4, true),
            ),
          }
        }
        return {
          format: part.input_audio.format,
          sampleRate: part.input_audio.sample_rate,
          samples: Array.from(part.input_audio.data),
        }
      }),
    )
    expect(inputs).toEqual([
      { format: 'pcm_f32', sampleRate: 16000, samples: [...samples] },
      { format: 'wav', sampleRate: 16000, samples: [...samples] },
      { format: 'wav', sampleRate: 16000, samples: [...samples] },
    ])
  },
)

test('Hermes keeps unsupported tool audio out of its next native request', async ({ page }) => {
  const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const endpoint = window.checks.engine().chat.completions
    Object.assign(window, { hermesAudioRequests: [] })
    endpoint.create = new Proxy(endpoint.create, {
      apply(target, self, args) {
        window.hermesAudioRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
  const data = 'UklGRjQAAABXQVZFZm10IBAAAAADAAEAgD4AAAD6AAAEACAAZmFjdAQAAAABAAAAZGF0YQQAAAAAAAAA'
  const paths: string[] = []
  const readFile = tool(
    ({ file_path }) => {
      paths.push(file_path)
      return [
        { type: 'text', text: `File: ${file_path}` },
        { type: 'audio', data, mimeType: 'audio/wav' },
      ]
    },
    {
      name: 'read_file',
      description: 'Read a WAV file and return its path and audio.',
      schema: z.object({ file_path: z.string() }),
    },
  )
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { max_tokens: 4096, temperature: 0.7, top_p: 0.9, seed: 7 },
  })
  const input = new HumanMessage(
    'Use read_file exactly once for /sound.wav, then reply with its file path.',
  )
  const call = await model.bindTools([readFile]).invoke([input])
  expect(call.tool_calls).toHaveLength(1)
  const original = await readFile.invoke({ ...call.tool_calls![0]!, type: 'tool_call' })
  if (!ToolMessage.isInstance(original)) throw new Error('Expected a tool message.')
  const messages = scrubUnsupportedMultimodalContent([input, call, original], model)
  // Complete with the bare model after executing WebLLM's native tool call.
  const answer = await model.invoke(messages)
  expect(paths).toEqual(['/sound.wav'])
  expect(answer.text).toContain('/sound.wav')
  expect(original.contentBlocks).toContainEqual(
    expect.objectContaining({ type: 'audio', data, mimeType: 'audio/wav' }),
  )
  const requests = await page.evaluate(() => window.hermesAudioRequests)
  expect(requests).toHaveLength(2)
  const parts = requests.flatMap((request) =>
    request.messages.flatMap((message) => (Array.isArray(message.content) ? message.content : [])),
  )
  expect(parts.some((part) => part.type === 'input_audio')).toBe(false)
  expect(requests[1]?.messages).toContainEqual({
    role: 'tool',
    tool_call_id: original.tool_call_id,
    content: expect.stringContaining('was not attached because this model does not support audio'),
  })
})

declare global {
  interface Window {
    audioRequests: ChatCompletionRequestStreaming[]
    hermesAudioRequests: ChatCompletionRequestStreaming[]
  }
}
