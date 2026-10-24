import { ModelType, prebuiltAppConfig } from '@mlc-ai/web-llm'
import { MemoryManager } from '@strands-agents/sdk'
import { InMemoryStorage, LocalFileStorage } from '@strands-agents/sdk/storage'
import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
import { BrowserModel, WebLLMEmbeddingSearchStrategy, type WebLLMChunk } from 'strands-web-llm'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test } from '../../../fixtures/browser.js'
import { createEmbeddingMemory, embeddingModelId } from '../../src/memory.js'
import { finalText, openHarness } from '../../fixtures/harness.js'

declare global {
  interface Window {
    memoryVectorReuses: number
  }
}

test('memory example keeps real vectors after queued work reuses the native arrays', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id, 512), embeddingModelId)
  const dir = await mkdtemp(join(tmpdir(), 'web-llm-memory-'))
  try {
    await page.evaluate(() =>
      window.webllmBridge.withEngine((engine) => {
        const create = engine.embeddings.create.bind(engine.embeddings)
        const buffers: number[][] = []
        window.memoryVectorReuses = 0
        engine.embeddings.create = async (request) => {
          const response = await create(request)
          const data = response.data.map((item) => {
            const buffer = (buffers[item.index] ??= [])
            buffer.splice(0, buffer.length, ...item.embedding)
            return { ...item, embedding: buffer }
          })
          void window.webllmBridge.withEngine(() => {
            for (const buffer of buffers) buffer.fill(0)
            window.memoryVectorReuses++
          })
          return { ...response, data }
        }
      }),
    )
    const memory = createEmbeddingMemory(page, 'gemma-4-E2B-it-q4f16_1-MLC', 8192, dir)
    const content = 'A basket of fresh apples.'
    await memory.add(content)
    const results = await memory.search('fruit')
    expect(
      await page.evaluate(() => window.webllmBridge.withEngine(() => window.memoryVectorReuses)),
    ).toBe(2)
    expect(results).toHaveLength(1)
    expect(results[0]?.content).toBe(content)
    expect(Number(results[0]?.metadata?.score)).toBeGreaterThan(0)
  } finally {
    try {
      await page.evaluate(() => window.harness.close())
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  }
})

for (const { model_id: modelId } of prebuiltAppConfig.model_list.filter(
  (model) => model.model_type === ModelType.embedding,
)) {
  test(`${modelId}: FileMemoryStore reuses real WebLLM vectors after copies and renames`, async ({
    page,
  }) => {
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id, 512), modelId)
    try {
      let documents = 0
      let queries = 0
      const strategy = new WebLLMEmbeddingSearchStrategy({
        includeData: false,
        embed: async (texts, type) => {
          if (type === 'document') documents += texts.length
          else queries += texts.length
          return page.evaluate(
            ({ input, model }) =>
              window.webllmBridge.withEngine(async (engine) => {
                const result = await engine.embeddings.create({
                  model,
                  input,
                  encoding_format: 'float',
                })
                return [...result.data]
                  .sort((a, b) => a.index - b.index)
                  .map(({ embedding }) => embedding)
              }),
            {
              model: modelId,
              input: texts.map((text) =>
                type === 'query'
                  ? `Represent this sentence for searching relevant passages: ${text}`
                  : text,
              ),
            },
          )
        },
      })
      const root = new InMemoryStorage({ searchStrategy: strategy })
      const memory = new FileMemoryStore({ name: 'notes', storage: root })
      const storage = root.namespace('memory/notes')
      const content = 'A basket of fresh apples.'
      const key = await memory.add(content)
      expect(await root.list('')).toEqual([`memory/notes/${key}`])
      const first = await memory.search('fruit')
      expect(first).toHaveLength(1)
      const score = Number(first[0]!.metadata?.score)
      expect(score).toBeGreaterThan(0)
      const data = new TextEncoder().encode(content)
      await storage.write('copy.md', data)
      expect(await memory.search('fruit')).toEqual(
        [key, 'copy.md'].map((path) => ({ content, metadata: { path, score } })),
      )
      await storage.delete(key)
      await storage.delete('copy.md')
      await storage.write('moved.md', data)
      expect(await memory.search('fruit')).toEqual([
        { content, metadata: { path: 'moved.md', score } },
      ])
      expect(documents).toBe(2)
      expect(queries).toBe(3)
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}

test('Strands search recovers after a dimension error and owns cached vectors', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id, 512), embeddingModelId)
  try {
    const storage = new InMemoryStorage()
    await storage.write('note', new TextEncoder().encode('A basket of fresh apples.'))
    let documentCalls = 0
    let queryVector: number[] = []
    let documentVector: number[] = []
    const strategy = new WebLLMEmbeddingSearchStrategy({
      minScore: -1,
      embed: async (texts, type) => {
        const vectors = await page.evaluate(
          ({ input, model }) =>
            window.webllmBridge.withEngine(async (engine) => {
              const response = await engine.embeddings.create({
                model,
                input,
                encoding_format: 'float',
              })
              return [...response.data]
                .sort((left, right) => left.index - right.index)
                .map(({ embedding }) => embedding)
            }),
          {
            model: embeddingModelId,
            input: texts.map((text) =>
              type === 'query'
                ? `Represent this sentence for searching relevant passages: ${text}`
                : text,
            ),
          },
        )
        if (type === 'query') queryVector = vectors[0]!
        else {
          documentCalls++
          if (documentCalls === 1) return vectors.map((vector) => vector.slice(0, -1))
          documentVector = vectors[0]!
        }
        return vectors
      },
    })
    await expect(strategy.search(storage, 'fruit')).rejects.toThrow('same dimensions')
    const results = await strategy.search(storage, 'fruit')
    expect(documentCalls).toBe(2)
    expect(queryVector).toHaveLength(384)
    expect(documentVector).toHaveLength(384)
    const norm = (vector: number[]) =>
      Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
    const dot = documentVector.reduce((sum, value, index) => sum + value * queryVector[index]!, 0)
    expect(results).toHaveLength(1)
    expect(results[0]!.key).toBe('note')
    expect(results[0]!.score).toBeCloseTo(dot / (norm(queryVector) * norm(documentVector)), 8)
    expect(await strategy.search(storage, 'fruit')).toEqual(results)
    documentVector.fill(0)
    expect(await strategy.search(storage, 'fruit')).toEqual(results)
    expect(documentCalls).toBe(2)
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

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
