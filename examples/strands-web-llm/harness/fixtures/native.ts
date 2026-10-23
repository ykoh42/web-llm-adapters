import { writeFile } from 'node:fs/promises'
import type { Page, TestInfo } from '@playwright/test'
import type { ChatCompletionChunk, ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'

/** Replay prepared requests without the adapter's formatter or output parser. */
export async function replayNative(page: Page, requests: ChatCompletionRequestStreaming[]) {
  return page.evaluate(
    (requests) =>
      window.webllmBridge.withEngine(async (engine) => {
        const outputs: ChatCompletionChunk[][] = []
        for (const request of requests) {
          await engine.resetChat()
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
