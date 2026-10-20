import { expect, it, vi } from 'vitest'
import { MLCEngine, type CompletionCreateParamsBase } from '@mlc-ai/web-llm'
import { protocolStream } from '../../shared/web-llm/protocol.js'

it('rebuilds Gemma request payloads after a failed open mutates its input', async () => {
  const engine = new MLCEngine({ logLevel: 'SILENT' })
  const payloads: CompletionCreateParamsBase[] = []
  const create = vi.spyOn(engine.completions, 'create').mockImplementation(async (request) => {
    payloads.push(structuredClone(request))
    if (payloads.length === 1) {
      request.prompt = 'Mutated by the failed engine call'
      request.extra_body!.enable_latency_breakdown = false
      throw new Error('Temporary opening failure')
    }
    return (async function* () {
      yield {
        id: 'raw',
        object: 'text_completion' as const,
        created: 0,
        model: request.model!,
        choices: [{ index: 0, text: '42', finish_reason: 'stop' as const, logprobs: null }],
      }
    })()
  })
  const response = protocolStream({
    engine,
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    history: [{ role: 'user', content: 'What is 6 * 7?' }],
    tools: [],
    choice: undefined,
    request: { extra_body: { enable_thinking: true, enable_latency_breakdown: true } },
    open: async (run) => {
      try {
        return await run()
      } catch {
        return run()
      }
    },
  })
  const output = []
  for await (const chunk of response) output.push(chunk)
  expect(create).toHaveBeenCalledTimes(2)
  expect(payloads[1]).toEqual(payloads[0])
  expect(output.at(-1)?.stop).toBe('endTurn')
})
