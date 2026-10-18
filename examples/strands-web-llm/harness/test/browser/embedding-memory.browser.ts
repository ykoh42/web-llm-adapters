import { MemoryManager } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { createEmbeddingMemory, embeddingModelId } from '../../src/memory.js'
import { finalText, openHarness } from '../../fixtures/harness.js'

test('WebLLM embeddings rank Strands memory and Harness injects the best match', async ({
  page,
}) => {
  const harness = await openHarness(page)
  const chatModelId = harness.model.modelId!

  try {
    await page.evaluate(
      ({ chatModelId, embeddingModelId }) =>
        window.webllmBridge.withEngine((engine) =>
          engine.reload([chatModelId, embeddingModelId], [{ context_window_size: 16384 }, {}]),
        ),
      { chatModelId, embeddingModelId },
    )

    const memory = createEmbeddingMemory(page, chatModelId, 16384, `${harness.dir}/semantic-memory`)
    await memory.add('Snowflake product\nThe Data Cloud!')
    await memory.add('Mexico City food\nMexico City of Course!')

    const results = await memory.search('what is snowflake?')
    expect(results[0]?.content).toContain('The Data Cloud!')
    expect(Number(results[0]?.metadata?.score)).toBeGreaterThan(Number(results[1]?.metadata?.score))

    const agent = await harness.create({
      memory: new MemoryManager({ stores: [memory], injection: { maxEntries: 1 } }),
    })
    const promptStart = await page.evaluate(() => window.gemmaPrompts.length)
    const answer = await agent.invoke('What is Snowflake?', { limits: { turns: 2 } })
    const prompt = (await page.evaluate(() => window.gemmaPrompts))[promptStart]

    expect(prompt).toContain('The Data Cloud!')
    expect(prompt).not.toContain('Mexico City of Course!')
    expect(finalText(answer)).toContain('Data Cloud')
  } finally {
    await harness.close()
  }
})
