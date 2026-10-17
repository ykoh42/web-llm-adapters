import { FunctionTool } from '@strands-agents/sdk'
import { test, expect } from '../../../test/browser.js'
import { openHarness, useTool, finalText } from './setup.js'

test('Harness tools share WebLLM embeddings, raw completions and resumable generation', async ({
  page,
}) => {
  const harness = await openHarness(page)
  const modelId = harness.model.modelId!
  const embeddingId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  try {
    await page.evaluate(
      ({ modelId, embeddingId }) =>
        window.webllmBridge.withEngine(async (engine) => {
          await engine.reload([modelId, embeddingId], [{ context_window_size: 16384 }, {}])
        }),
      { modelId, embeddingId },
    )

    let dimensions = 0
    const vectorSize = new FunctionTool({
      name: 'vector_size',
      description: 'Embed text and return its vector size.',
      inputSchema: {
        type: 'object',
        properties: { text: { type: 'string', description: 'Text to embed' } },
        required: ['text'],
        additionalProperties: false,
      },
      callback: async (input) => {
        const { text } = input as { text: string }
        const vector = await page.evaluate(
          ({ model, text }) =>
            window.webllmBridge.withEngine(async (engine) => {
              const result = await engine.embeddings.create({ model, input: text })
              return result.data[0]!.embedding
            }),
          { model: embeddingId, text },
        )
        expect(vector.every(Number.isFinite)).toBe(true)
        dimensions = vector.length
        return { dimensions }
      },
    })
    const agent = await harness.create({ tools: [vectorSize] })
    useTool(agent, 'vector_size')
    const [answer, raw] = await Promise.all([
      agent.invoke(
        'Use vector_size to embed the text hello. Tell me the number of dimensions it returned.',
        {
          limits: { turns: 3 },
        },
      ),
      page.evaluate(
        (model) =>
          window.webllmBridge.withEngine((engine) =>
            engine.completions.create({
              model,
              prompt: 'A short greeting:',
              n: 2,
              max_tokens: 16,
              temperature: 0.7,
              seed: 7,
            }),
          ),
        modelId,
      ),
    ])
    expect(dimensions).toBe(384)
    expect(finalText(answer)).toContain('384')
    expect(raw.choices).toHaveLength(2)
    expect(raw.choices.every((choice) => choice.text.length > 0)).toBe(true)

    const resumed = await page.evaluate(
      (model) =>
        window.webllmBridge.withEngine(async (engine) => {
          const sessionId = crypto.randomUUID()
          try {
            const stream = await engine.chat.completions.create({
              model,
              messages: [{ role: 'user', content: 'Count from 1 to 30. Only numbers.' }],
              stream: true,
              max_tokens: 128,
              temperature: 0,
              extra_body: { resumable: { enabled: true, sessionId, checkpointIntervalTokens: 8 } },
            })
            let partial = ''
            let count = 0
            for await (const chunk of stream) {
              const text = chunk.choices[0]?.delta.content
              if (!text) continue
              partial += text
              if (++count === 8) engine.interruptGenerate()
            }
            const sessions = await engine.listResumableSessions()
            const saved = await engine.resumeChatCompletion(sessionId)
            if (!('recoveredText' in saved)) throw new Error('Expected a restored snapshot')
            const continued = await engine.resumeChatCompletion(sessionId, {
              continueGeneration: true,
              stream: true,
            })
            if ('recoveredText' in continued) throw new Error('Expected a resumed stream')
            let text = ''
            for await (const chunk of continued) text += chunk.choices[0]?.delta.content ?? ''
            return {
              partial,
              text,
              saved,
              probe: sessions.find((session) => session.sessionId === sessionId),
            }
          } finally {
            await engine.deleteResumableSession(sessionId)
          }
        }),
      modelId,
    )
    expect(resumed.probe?.resumable).toBe(true)
    expect(resumed.probe?.recoveryMode).toMatch(/^(kv|token_replay)$/)
    expect(resumed.partial.length).toBeGreaterThan(0)
    expect(resumed.saved.recoveredText.length).toBeGreaterThan(0)
    // Inspecting a snapshot only restores text; continuation restores generation state.
    expect(resumed.saved.recoveryMode).toBe('text_only')
    expect(resumed.text.length).toBeGreaterThan(0)
    expect(resumed.saved.recoveredText + resumed.text).toContain('30')
    const after = await agent.invoke('Reply with the word ready.', { limits: { turns: 3 } })
    expect(finalText(after)).toMatch(/ready/i)
    expect(agent.messages.flatMap((message) => message.content)).toContainEqual(
      expect.objectContaining({ type: 'toolUseBlock', name: 'vector_size' }),
    )
  } finally {
    await harness.close()
  }
})
