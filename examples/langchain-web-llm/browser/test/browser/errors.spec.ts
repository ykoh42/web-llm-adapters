import { writeFile } from 'node:fs/promises'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

type NativeRequest = ChatCompletionRequestStreaming | CompletionCreateParamsStreaming

test('Worker failures preserve native causes, callback edits and engine recovery', async ({
  page,
}, info) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    window.errorCauseRequests = []
    const engine = window.checks.engine()
    for (const endpoint of [engine.chat.completions, engine.completions]) {
      endpoint.create = new Proxy(endpoint.create, {
        apply(target, self, args) {
          window.errorCauseRequests.push(structuredClone(args[0]))
          return Reflect.apply(target, self, args)
        },
      })
    }
  })
  async function direct(request: NativeRequest) {
    return page.evaluate(
      (request) =>
        window.webllmLangChainBridge!.withEngine(async (engine) => {
          try {
            const chunks =
              'messages' in request
                ? await engine.chat.completions.create(request)
                : await engine.completions.create(request)
            for await (const chunk of chunks) void chunk
            return { type: 'none', value: '' }
          } catch (error) {
            return { type: typeof error, value: String(error) }
          }
        }),
      request,
    )
  }
  let attempts = 0
  const details: unknown = JSON.parse(
    '{"__proto__":{"kept":true},"items":[{"__proto__":{"nested":true},"value":42}]}',
  )
  const openingModel = new BrowserChatWebLLM(page, {
    modelId,
    maxRetries: 0,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 0.5 },
    onFailedAttempt(error) {
      attempts++
      expect(typeof error.cause).toBe('string')
      error.cause = { native: error.cause, checked: true, details }
      error.metadata = details
    },
  })
  const opening = await openingModel.invoke('Hello').catch((error: unknown) => error)
  if (!(opening instanceof Error)) throw opening
  const openingRequest = await page.evaluate(() => window.errorCauseRequests.at(-1)!)
  const openingNative = await direct(openingRequest)
  expect(openingNative.type).toBe('string')
  expect(openingNative.value.toLowerCase()).toContain('seed')
  expect(attempts).toBe(1)
  expect(opening.message).toBe(openingNative.value)
  expect(opening.cause).toStrictEqual({
    native: openingNative.value,
    checked: true,
    details,
  })
  expect(opening).toHaveProperty('metadata', details)

  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const generation = await model
    .invoke('Hello', { request: { temperature: -1 } })
    .catch((error: unknown) => error)
  if (!(generation instanceof Error)) throw generation
  const generationRequest = await page.evaluate(() => window.errorCauseRequests.at(-1)!)
  const generationNative = await direct(generationRequest)
  expect(generationNative.type).toBe('string')
  expect(generationNative.value).toContain('temperature')
  expect(generation).toMatchObject({
    message: generationNative.value,
    cause: generationNative.value,
  })

  const answer = await model.invoke('What is 17 * 23?')
  expect(answer.text).toContain('391')
  expect(answer.contentBlocks.some((block) => block.type === 'reasoning' && block.reasoning)).toBe(
    true,
  )
  await writeFile(
    info.outputPath('native-causes.json'),
    JSON.stringify(
      {
        opening: { message: opening.message, cause: opening.cause },
        openingNative,
        generation: { message: generation.message, cause: generation.cause },
        generationNative,
        answer,
      },
      null,
      2,
    ),
  )
})

declare global {
  interface Window {
    errorCauseRequests: NativeRequest[]
  }
}
