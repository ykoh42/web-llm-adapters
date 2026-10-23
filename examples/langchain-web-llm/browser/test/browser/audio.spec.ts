import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { scrubUnsupportedMultimodalContent } from 'deepagents'
import { z } from 'zod'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'

test('Float32 PCM matches native WebLLM through local and browser LangChain models', async ({
  page,
}) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const prompt = 'Describe any sound you hear in one short sentence.'
  const request = { temperature: 0, top_p: 1, repetition_penalty: 1, max_tokens: 4096, seed: 7 }
  const samples = Float32Array.from(
    { length: 16000 },
    (_, index) => 0.1 * Math.sin((2 * Math.PI * 440 * index) / 16000),
  )
  const native = await page.evaluate(
    async ({ prompt, samples, request, modelId }) => {
      const engine = window.checks.engine()
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
      let text = ''
      let usage
      for await (const chunk of await engine.chat.completions.create({
        ...request,
        model: modelId,
        messages,
        n: 1,
        stream: true,
        stream_options: { include_usage: true },
      })) {
        text += chunk.choices[0]?.delta.content ?? ''
        usage = chunk.usage ?? usage
      }
      await engine.resetChat()
      const local = await window.checks.model(false).invoke(messages, { request })
      return { text, usage, local: { text: local.text, usage: local.usage_metadata } }
    },
    { prompt, samples: [...samples], request, modelId },
  )

  await page.evaluate(() => window.checks.engine().resetChat())
  const model = new BrowserChatWebLLM(page, { modelId, request })
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

  expect(native.text.length).toBeGreaterThan(0)
  expect(native.usage?.prompt_tokens).toBeGreaterThan(0)
  for (const result of [native.local, { text: remote.text, usage: remote.usage_metadata }]) {
    expect(result.text).toBe(native.text)
    expect(result.usage?.input_tokens).toBe(native.usage?.prompt_tokens)
    expect(result.usage?.output_tokens).toBe(native.usage?.completion_tokens)
  }
})

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
    hermesAudioRequests: ChatCompletionRequestStreaming[]
  }
}
