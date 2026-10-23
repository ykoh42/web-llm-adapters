import { writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { BrowserModel } from 'strands-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { openHarness, useTool, finalText } from '../../fixtures/harness.js'

test('Harness model latency matches native WebLLM across a thinking and tool loop', async ({
  page,
}) => {
  const harness = await openHarness(page)
  let latencyMs = 0
  let calls = 0
  const model = new BrowserModel(page, {
    ...harness.model.getConfig(),
    onChunk: (chunk) => {
      if (!chunk.usage) return
      latencyMs += chunk.usage.extra.e2e_latency_s * 1000
      calls++
    },
  })
  try {
    const code = crypto.randomUUID().slice(0, 8)
    const path = join(harness.dir, 'code.txt')
    await writeFile(path, `The secret code is ${code}.`)
    const agent = await harness.create({ model })
    useTool(agent, 'read')
    const result = await agent.invoke(
      `Use read to read ${path}. Reply with the secret code from that file.`,
      { limits: { turns: 4 } },
    )

    expect(finalText(result)).toContain(code)
    expect(agent.messages.flatMap((message) => message.content)).toContainEqual(
      expect.objectContaining({ type: 'toolResultBlock', status: 'success' }),
    )
    expect(
      agent.messages
        .flatMap((message) => message.content)
        .some((block) => block.type === 'reasoningBlock' && block.text?.trim()),
    ).toBe(true)
    expect(calls).toBeGreaterThanOrEqual(2)
    expect(latencyMs).toBeGreaterThan(0)
    expect(result.metrics?.accumulatedMetrics.latencyMs).toBeCloseTo(latencyMs, 6)
  } finally {
    await harness.close()
  }
})
