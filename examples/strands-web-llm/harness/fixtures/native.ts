import { writeFile } from 'node:fs/promises'
import type { Page, TestInfo } from '@playwright/test'
import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'

/** Replay prepared requests without the adapter's formatter or output parser. */
export async function replayNative(page: Page, requests: ChatCompletionRequestStreaming[]) {
  return page.evaluate(
    (requests) =>
      window.webllmBridge.withEngine(async (engine) => {
        const outputs: ChatCompletionChunk[][] = []
        await engine.resetChat()
        for (const request of requests) {
          const chunks: ChatCompletionChunk[] = []
          for await (const chunk of await engine.chat.completions.create(request))
            chunks.push(chunk)
          outputs.push(chunks)
        }
        return outputs
      }),
    requests,
  )
}

/** Keep native evidence on disk, including when using the line reporter. */
export async function recordNative(info: TestInfo, name: string, data: unknown) {
  const path = info.outputPath(`${name}.json`)
  await writeFile(path, JSON.stringify(data))
  await info.attach(name, { path, contentType: 'application/json' })
}

/** Observe the chat requests made by the running browser engine. */
export async function captureChatRequests(page: Page) {
  await page.evaluate(() =>
    window.webllmBridge.withEngine((engine) => {
      window.releaseChatRequests = []
      const endpoint = engine.chat.completions
      endpoint.create = new Proxy(endpoint.create, {
        apply(target, self, args) {
          window.releaseChatRequests.push(structuredClone(args[0]))
          return Reflect.apply(target, self, args)
        },
      })
    }),
  )
}

declare global {
  interface Window {
    releaseChatRequests: ChatCompletionRequestStreaming[]
  }
}
