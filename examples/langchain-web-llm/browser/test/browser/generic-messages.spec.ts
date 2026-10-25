import { AIMessage, ChatMessage, HumanMessage, SystemMessage } from '@langchain/core/messages'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

declare global {
  interface Window {
    genericMessageRequests: (ChatCompletionRequestStreaming | CompletionCreateParamsStreaming)[]
  }
}

for (const modelId of ['gemma-4-E2B-it-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  test(`${modelId}: generic roles keep the canonical request and real thinking answer`, async ({
    page,
  }, info) => {
    await page.evaluate((id) => window.checks.load(id, true), modelId)
    await page.evaluate(() => {
      window.genericMessageRequests = []
      const engine = window.checks.engine()
      for (const endpoint of [engine.chat.completions, engine.completions]) {
        endpoint.create = new Proxy(endpoint.create, {
          apply(target, self, args) {
            window.genericMessageRequests.push(structuredClone(args[0]))
            return Reflect.apply(target, self, args)
          },
        })
      }
    })
    const chunks: WebLLMChunk[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => chunks.push(chunk),
    })
    const canonical = [
      new SystemMessage('Answer accurately.'),
      new HumanMessage('What is 3 + 3?'),
      new AIMessage('6'),
      new HumanMessage('What is 17 * 23?'),
    ]
    const generic = [
      new ChatMessage('Answer accurately.', 'system'),
      new ChatMessage('What is 3 + 3?', 'user'),
      new ChatMessage('6', 'assistant'),
      new ChatMessage('What is 17 * 23?', 'user'),
    ]
    const aliases = [
      new ChatMessage('Answer accurately.', 'system'),
      new ChatMessage('What is 3 + 3?', 'human'),
      new ChatMessage('6', 'ai'),
      new ChatMessage('What is 17 * 23?', 'human'),
    ]
    const responses = []
    for (const history of [canonical, generic, aliases]) {
      chunks.length = 0
      const response = await model.invoke(history)
      expect(response.text).toContain('391')
      expect(
        response.contentBlocks.some(
          (block) => block.type === 'reasoning' && block.reasoning.trim(),
        ),
      ).toBe(true)
      expect(chunks.at(-1)?.usage?.completion_tokens).toBeGreaterThan(0)
      const text = chunks
        .flatMap((chunk) =>
          chunk.choices.map((choice) =>
            'text' in choice ? choice.text : (choice.delta.content ?? ''),
          ),
        )
        .join('')
      expect(text).toContain('391')
      responses.push({ content: response.content, native: chunks.slice() })
    }
    const requests = await page.evaluate(() => window.genericMessageRequests)
    expect(requests).toHaveLength(3)
    expect(requests[1]).toEqual(requests[0])
    expect(requests[2]).toEqual(requests[0])
    await info.attach('generic-message-parity', {
      body: JSON.stringify({ requests, responses }),
      contentType: 'application/json',
    })
  })
}
