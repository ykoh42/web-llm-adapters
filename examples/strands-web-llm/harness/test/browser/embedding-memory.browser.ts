import { MemoryManager } from '@strands-agents/sdk'
import { LocalFileStorage } from '@strands-agents/sdk/storage'
import { BrowserModel, WebLLMEmbeddingSearchStrategy, type WebLLMChunk } from 'strands-web-llm'
import { writeFile } from 'node:fs/promises'
import { expect, test } from '../../../fixtures/browser.js'
import { createEmbeddingMemory, embeddingModelId } from '../../src/memory.js'
import { finalText, openHarness } from '../../fixtures/harness.js'

test('separate Strands stores reuse their real WebLLM document vectors', async ({ page }) => {
  const harness = await openHarness(page)
  let documentCalls = 0
  try {
    await page.evaluate(
      (id) => window.webllmBridge.withEngine((engine) => engine.reload(id)),
      embeddingModelId,
    )
    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => {
        if (type === 'document') documentCalls++
        return page.evaluate(
          ({ texts, model }) =>
            window.webllmBridge.withEngine(async (engine) => {
              const response = await engine.embeddings.create({
                model,
                input: texts,
                encoding_format: 'float',
              })
              return [...response.data]
                .sort((left, right) => left.index - right.index)
                .map(({ embedding }) => embedding)
            }),
          {
            texts: texts.map((text) =>
              type === 'query'
                ? `Represent this sentence for searching relevant passages: ${text}`
                : text,
            ),
            model: embeddingModelId,
          },
        )
      },
    })
    const storage = new LocalFileStorage(`${harness.dir}/notes`, { searchStrategy: strategy })
    const first = storage.namespace('first')
    const second = storage.namespace('second')
    const encoder = new TextEncoder()
    const decoder = new TextDecoder()
    await first.write('note', encoder.encode('The Data Cloud!'))
    await second.write('note', encoder.encode('A bread recipe.'))
    for (let round = 0; round < 3; round++) {
      expect(decoder.decode((await first.search('cloud'))[0]?.data)).toBe('The Data Cloud!')
      expect(decoder.decode((await second.search('bread'))[0]?.data)).toBe('A bread recipe.')
    }
    await first.delete('note')
    expect(await first.search('cloud')).toEqual([])
    expect(decoder.decode((await second.search('bread'))[0]?.data)).toBe('A bread recipe.')
    expect(documentCalls).toBe(2)
  } finally {
    await harness.close()
  }
})

test('batches large WebLLM embedding searches', async ({ page }) => {
  const harness = await openHarness(page)
  const documentBatchSizes: number[] = []
  let queryCalls = 0

  try {
    await page.evaluate(
      (model) => window.webllmBridge.withEngine((engine) => engine.reload(model)),
      embeddingModelId,
    )
    const storage = new LocalFileStorage(`${harness.dir}/large-store`)
    const encoder = new TextEncoder()
    const keys = Array.from({ length: 65 }, (_, index) => `note-${index}`)
    for (const key of keys) await storage.write(key, encoder.encode(`Short note ${key}.`))

    const strategy = new WebLLMEmbeddingSearchStrategy({
      embed: async (texts, type) => {
        if (type === 'query') queryCalls++
        else documentBatchSizes.push(texts.length)
        return page.evaluate(
          ({ texts, type, model }) =>
            window.webllmBridge.withEngine(async (engine) => {
              const input = texts.map((text) =>
                type === 'query'
                  ? `Represent this sentence for searching relevant passages: ${text}`
                  : text,
              )
              const result = await engine.embeddings.create({ model, input })
              return [...result.data]
                .sort((left, right) => left.index - right.index)
                .map(({ embedding }) => embedding)
            }),
          { texts: [...texts], type, model: embeddingModelId },
        )
      },
    })

    const results = await strategy.search(storage, 'note 42')

    expect(queryCalls).toBe(1)
    expect(documentBatchSizes).toEqual([32, 32, 1])
    expect(results.map(({ key }) => key).sort()).toEqual(keys.sort())
    expect(results.every(({ score }) => Number.isFinite(score))).toBe(true)
  } finally {
    await harness.close()
  }
})

test('WebLLM embeddings rank Strands memory and Harness injects the best match', async ({
  page,
}, info) => {
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

    const raw: WebLLMChunk[] = []
    const agent = await harness.create({
      model: new BrowserModel(page, {
        ...harness.model.getConfig(),
        onChunk: (chunk) => raw.push(chunk),
      }),
      memory: new MemoryManager({ stores: [memory], injection: { maxEntries: 1 } }),
    })
    const promptStart = await page.evaluate(() => window.gemmaPrompts.length)
    const answer = await agent.invoke('What is Snowflake?')
    const path = info.outputPath('inference.json')
    await writeFile(path, JSON.stringify({ result: answer, messages: agent.messages, raw }))
    await info.attach('inference', {
      path,
      contentType: 'application/json',
    })
    const prompt = (await page.evaluate(() => window.gemmaPrompts))[promptStart]

    expect(prompt).toContain('The Data Cloud!')
    expect(prompt).not.toContain('Mexico City of Course!')
    expect(answer.stopReason).toBe('endTurn')
    expect(finalText(answer)).toContain('Data Cloud')
  } finally {
    await harness.close()
  }
})
