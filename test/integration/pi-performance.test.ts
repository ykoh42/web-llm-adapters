import { expect, it, vi } from 'vitest'
import { createModels } from '@earendil-works/pi-ai/models'
import { createWebLLMProvider } from 'pi-ai-web-llm'
import type { ChatCompletionRequestBase } from '@mlc-ai/web-llm'
import { chunk, chunks, setupEngine } from '../../fixtures/web-llm.js'
import { comparePerformance } from '../../examples/pi-ai-web-llm/browser/test/performance.js'

it('compares the same native request and output in both directions', async () => {
  const output = {
    ...chunk({ content: 'A short story.' }, 'stop'),
    usage: {
      prompt_tokens: 2,
      completion_tokens: 4,
      total_tokens: 6,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 1,
        decode_tokens_per_s: 1,
        time_to_first_token_s: 1,
        time_per_output_token_s: 1,
      },
    },
  }
  const { engine, create } = setupEngine(output)
  const requests: ChatCompletionRequestBase[] = []
  create.mockImplementation(async (request) => {
    requests.push(structuredClone(request))
    request.messages.push({ role: 'user', content: 'Engine mutation' })
    return chunks(output)
  })
  const reset = vi.spyOn(engine, 'resetChat').mockResolvedValue()
  const models = createModels()
  const provider = createWebLLMProvider({
    engine,
    modelId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    contextWindow: 4096,
    request: { max_tokens: 128, temperature: 0, seed: 7 },
  })
  models.setProvider(provider)
  const samples = await comparePerformance(models, provider.getModels()[0]!, engine)

  expect(samples).toHaveLength(4)
  expect(reset).toHaveBeenCalledTimes(10)
  expect(create).toHaveBeenCalledTimes(10)
  for (const request of requests) expect(request).toEqual(requests[0])
  for (const { native, adapter } of samples) {
    expect(native).toMatchObject({ text: 'A short story.', tokens: 4 })
    expect(adapter).toMatchObject({ text: 'A short story.', tokens: 4 })
  }
})
