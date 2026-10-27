import { expect, it, vi } from 'vitest'
import { setupEngine, collect } from '../../fixtures/web-llm.js'
import { protocolStream } from '../../shared/web-llm/protocol.js'
import { withEngine } from '../../shared/web-llm/engine/access.js'

it.each([
  'native-model',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
])('%s does not open a native request aborted during preparation', async (modelId) => {
  const { engine, create } = setupEngine()
  const raw = vi
    .spyOn(engine.completions, 'create')
    .mockRejectedValue(new Error('Unexpected native call'))
  const controller = new AbortController()
  const reason = new Error('Cancelled during preparation')
  const stream = protocolStream({
    engine,
    modelId,
    history: [{ role: 'user', content: 'Hello' }],
    tools: [],
    choice: undefined,
    request: { extra_body: { enable_thinking: true } },
    signal: controller.signal,
    prepare: async (request) => {
      await Promise.resolve()
      controller.abort(reason)
      return request
    },
  })

  await expect(collect(stream)).rejects.toBe(reason)
  expect(create).not.toHaveBeenCalled()
  expect(raw).not.toHaveBeenCalled()
  await expect(withEngine(engine, () => 'Available')).resolves.toBe('Available')
})
