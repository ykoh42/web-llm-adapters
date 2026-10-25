import { AIMessage } from '@langchain/core/messages'
import { BaseCallbackHandler } from '@langchain/core/callbacks/base'
import { tool } from '@langchain/core/tools'
import { writeFile } from 'node:fs/promises'
import { createAgent } from 'langchain'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of [
  'Qwen2.5-3B-Instruct-q4f32_1-MLC',
  'Qwen3-4B-q4f32_1-MLC',
  'Qwen3.5-4B-q4f32_1-MLC',
]) {
  test(`${modelId} preserves native output through a complete tool loop`, async ({
    page,
  }, info) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const thinking = !modelId.startsWith('Qwen2.5-')
    const secret = 'secret-' + crypto.randomUUID().slice(0, 8)
    const inputs: { code: string }[] = []
    const lookup = tool(
      (input) => {
        inputs.push(input)
        return { secret }
      },
      {
        name: 'lookup',
        description: 'Look up a secret by its code.',
        schema: z.object({ code: z.string() }),
      },
    )
    const chunks: WebLLMChunk[] = []
    const originals: string[] = []
    const capture = BaseCallbackHandler.fromMethods({
      handleLLMEnd: async () => {
        originals.push(
          await page.evaluate(
            (id) => window.webllmLangChainBridge!.withEngine((engine) => engine.getMessage(id)),
            modelId,
          ),
        )
      },
    })
    capture.awaitHandlers = true
    capture.raiseError = true
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request: { ...sampling(modelId, thinking), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => chunks.push(chunk),
      callbacks: [capture],
    })
    const result = await createAgent({ model, tools: [lookup] })
      .invoke({
        messages: [
          { role: 'user', content: 'Use lookup with code A7, then reply with its secret.' },
        ],
      })
      .finally(async () => {
        const path = info.outputPath('native.json')
        await writeFile(path, JSON.stringify({ inputs, originals, chunks }))
        await info.attach('native', { path, contentType: 'application/json' })
      })
    expect(inputs.length).toBeGreaterThan(0)
    for (const input of inputs) expect(input).toEqual({ code: 'A7' })
    const messages = result.messages.filter(AIMessage.isInstance)
    expect(messages.at(-1)?.text).toContain(secret)
    expect(
      messages
        .flatMap((message) => message.contentBlocks)
        .some((block) => block.type === 'reasoning' && block.reasoning),
    ).toBe(thinking)
    const actual = new Map<string, string>()
    for (const chunk of chunks) {
      const text = chunk.choices
        .map((choice) => ('delta' in choice ? (choice.delta.content ?? '') : choice.text))
        .join('')
      actual.set(chunk.id, (actual.get(chunk.id) ?? '') + text)
    }
    expect([...actual.values()]).toEqual(originals)
    const final = originals.at(-1)!
    const thought = thinking
      ? (modelId.startsWith('Qwen3.5-')
          ? /^([\s\S]*?)<\/think>/
          : /^<think>([\s\S]*?)<\/think>/
        ).exec(final)
      : null
    expect(messages.at(-1)?.text).toBe(final.slice(thought?.[0].length ?? 0))
  })
}
