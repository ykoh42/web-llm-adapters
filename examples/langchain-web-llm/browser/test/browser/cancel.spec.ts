import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { getRetryable } from '@langchain/core/errors'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of [
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
]) {
  for (const phase of ['raw callback', 'token callback']) {
    test(`${modelId} cancels at ${phase} and runs the next request`, async ({ page }) => {
      await page.evaluate((id) => window.checks.load(id), modelId)
      const thinking = !modelId.startsWith('Hermes-')
      const controller = new AbortController()
      let callbacks = 0
      let atCancel = 0
      const model = new BrowserChatWebLLM(page, {
        modelId,
        request: { temperature: 0, ...sampling(modelId, thinking), max_tokens: 4096, seed: 7 },
        onChunk: (chunk) => {
          callbacks++
          const choice = chunk.choices[0]
          const text = choice && ('delta' in choice ? choice.delta.content : choice.text)
          if (phase === 'raw callback' && text && !controller.signal.aborted) {
            atCancel = callbacks
            controller.abort()
          }
        },
      })
      const bound = model.bindTools([
        {
          type: 'function',
          function: {
            name: 'lookup',
            description: 'Look up a code.',
            parameters: {
              type: 'object',
              properties: { code: { type: 'string' } },
              required: ['code'],
            },
          },
        },
      ])
      const output: string[] = []
      let tokenCallbacks = 0
      const onToken = () => {
        tokenCallbacks++
        if (phase === 'token callback') {
          atCancel = callbacks
          controller.abort()
        }
      }
      await expect(
        (async () => {
          for await (const chunk of await bound.stream('Use lookup to look up code A7.', {
            signal: controller.signal,
            callbacks: [{ handleLLMNewToken: onToken }],
          }))
            output.push(chunk.text)
        })(),
      ).rejects.toHaveProperty('name', 'AbortError')
      expect(atCancel).toBeGreaterThan(0)
      expect(callbacks).toBe(atCancel)
      expect(tokenCallbacks).toBe(phase === 'raw callback' ? 0 : 1)
      expect(output).toEqual([])
      const recovered = await bound.invoke('Use lookup to look up code A7.')
      expect(recovered.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
      if (thinking)
        expect(recovered.contentBlocks.some((block) => block.type === 'reasoning')).toBe(true)
    })
  }
}

test('Qwen2.5 buffered invoke cancels during a browser read', async ({ page }) => {
  const modelId = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const engine = window.checks.engine()
    window.bufferedRequestStarted = false
    for (const endpoint of [engine.chat.completions, engine.completions]) {
      Object.assign(endpoint, {
        create: new Proxy(endpoint.create, {
          apply(target, self, args) {
            window.bufferedRequestStarted = true
            return Reflect.apply(target, self, args)
          },
        }),
      })
    }
  })
  const controller = new AbortController()
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, false), max_tokens: 4096, seed: 7 },
  })
  const pending = model.invoke('Write a detailed story about a hidden island.', {
    signal: controller.signal,
  })

  await page.waitForFunction(() => window.bufferedRequestStarted)
  controller.abort()
  await expect(pending).rejects.toHaveProperty('name', 'AbortError')
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId} identifies a native interrupt as terminal cancellation`, async ({
    page,
  }, info) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(() => {
      const engine = window.checks.engine()
      let cancelled = false
      window.interruptRequests = []
      for (const endpoint of [engine.chat.completions, engine.completions]) {
        Object.assign(endpoint, {
          create: new Proxy(endpoint.create, {
            async apply(target, self, args) {
              if (args[0].stream !== true) throw new Error('This test needs streaming requests.')
              window.interruptRequests.push(structuredClone(args[0]))
              const source: AsyncIterable<WebLLMChunk> = await Reflect.apply(target, self, args)
              return (async function* () {
                for await (const chunk of source) {
                  const choice = chunk.choices[0]
                  const text = choice && ('delta' in choice ? choice.delta.content : choice.text)
                  if (text && !cancelled) {
                    cancelled = true
                    await engine.interruptGenerate()
                  }
                  yield chunk
                }
              })()
            },
          }),
        })
      }
    })
    const stops: string[] = []
    const raw: WebLLMChunk[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => {
        raw.push(chunk)
        const reason = chunk.choices[0]?.finish_reason
        if (reason) stops.push(reason)
      },
    })
    const failure = await model
      .invoke('Explain the history of mathematics in detail.')
      .catch((error: unknown) => error)
    expect(stops).toEqual(['abort'])
    expect(failure).toMatchObject({ name: 'AbortError', message: 'WebLLM cancelled generation.' })
    expect(getRetryable(failure)).toBe(false)

    stops.length = 0
    raw.length = 0
    const recovered = await model.invoke('What is 3 + 3? Reply briefly.')
    const native = await page.evaluate(async () => {
      const engine = window.checks.engine()
      const input = window.interruptRequests.at(-1)!
      const source =
        'prompt' in input
          ? await engine.completions.create(input)
          : await engine.chat.completions.create(input)
      const chunks: WebLLMChunk[] = []
      for await (const chunk of source) chunks.push(chunk)
      return { input, chunks }
    })
    const path = info.outputPath('native-recovery.json')
    await writeFile(
      path,
      JSON.stringify(
        {
          request: native.input,
          response: recovered.content,
          raw: nativeText(raw),
          native: nativeText(native.chunks),
        },
        null,
        2,
      ),
    )
    await info.attach('native recovery', { path, contentType: 'application/json' })
    expect(nativeText(raw)).toBe(nativeText(native.chunks))
    if ('prompt' in native.input) expect(native.input.prompt).toContain('<|think|>')
    else expect(native.input.extra_body?.enable_thinking).toBe(true)
    const tokens = native.chunks.filter((chunk) => chunk.usage).at(-1)?.usage?.completion_tokens
    expect(tokens).toBeGreaterThan(0)
    expect(recovered.usage_metadata?.output_tokens).toBe(tokens)
    expect(stops).toEqual(['stop'])
    expect(native.chunks.map((chunk) => chunk.choices[0]?.finish_reason).filter(Boolean)).toEqual([
      'stop',
    ])
    expect(recovered.text).toContain('6')
  })
}

function nativeText(chunks: WebLLMChunk[]) {
  return chunks
    .map((chunk) => {
      const choice = chunk.choices[0]
      if (!choice) return ''
      return 'delta' in choice ? (choice.delta.content ?? '') : choice.text
    })
    .join('')
}

declare global {
  interface Window {
    bufferedRequestStarted: boolean
    interruptRequests: (ChatCompletionRequestStreaming | CompletionCreateParamsStreaming)[]
  }
}
