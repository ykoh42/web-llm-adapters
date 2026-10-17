import { FunctionTool } from '@strands-agents/sdk'
import { join } from 'node:path'
import { test, expect } from '../../../fixtures/browser.js'
import { openHarness, useTool, finalText } from '../../fixtures/harness.js'

declare global {
  interface Window {
    checkpoint: { text: string; ready: boolean; error?: string }
  }
}

test('a new page restores native generation and Harness tool history independently', async ({
  page,
}) => {
  const harness = await openHarness(page)
  const modelId = harness.model.modelId!
  const request = harness.model.getConfig().request!
  harness.model.updateConfig({
    request: {
      ...request,
      extra_body: { ...request.extra_body, resumable: { enabled: false } },
    },
  })
  const sessionId = crypto.randomUUID()
  const session = { id: 'recovery', dir: join(harness.dir, 'sessions') }
  const code = 'tag-' + crypto.randomUUID().slice(0, 8)
  let calls = 0
  const tag = new FunctionTool({
    name: 'get_tag',
    description: 'Get the stored tag for a project.',
    inputSchema: {
      type: 'object',
      properties: { project: { type: 'string' } },
      required: ['project'],
    },
    callback: () => {
      calls++
      return { tag: code }
    },
  })
  try {
    const agent = await harness.create({ session, tools: [tag] })
    useTool(agent, 'get_tag')
    const answer = await agent.invoke('Get the tag for project demo. Reply with the tag.', {
      limits: { turns: 3 },
    })
    expect(finalText(answer)).toContain(code)
    expect(calls).toBe(1)
    await agent.shutdown()
    const history = agent.messages.map((message) => message.toJSON())
    expect(
      agent.messages.some((message) =>
        message.content.some((block) => block.type === 'reasoningBlock' && block.text),
      ),
    ).toBe(true)

    await page.evaluate(
      ({ model, sessionId }) => {
        window.checkpoint = { text: '', ready: false }
        void window.webllmBridge
          .withEngine(async (engine) => {
            const stream = await engine.chat.completions.create({
              model,
              messages: [{ role: 'user', content: 'Count from 1 to 30. Only numbers.' }],
              stream: true,
              max_tokens: 128,
              temperature: 0,
              extra_body: {
                resumable: {
                  enabled: true,
                  sessionId,
                  durabilityMode: 'exact',
                  strictPersistence: true,
                  checkpointIntervalTokens: 8,
                },
              },
            })
            let count = 0
            for await (const chunk of stream) {
              const text = chunk.choices[0]?.delta.content
              if (!text) continue
              window.checkpoint.text += text
              if (++count === 8) {
                window.checkpoint.ready = true
                // Leave generation open. Reloading the page destroys its worker without cleanup.
                await new Promise(() => {})
              }
            }
          })
          .catch((error: unknown) => {
            window.checkpoint.error = String(error)
          })
      },
      { model: modelId, sessionId },
    )
    await page.waitForFunction(() => window.checkpoint.ready || window.checkpoint.error)
    const checkpoint = await page.evaluate(() => window.checkpoint)
    expect(checkpoint.error).toBeUndefined()
    expect(checkpoint.text.length).toBeGreaterThan(0)
    await page.reload()
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((model) => window.harness.open(model, 16384), modelId)

    const recovered = await page.evaluate(
      (sessionId) =>
        window.webllmBridge.withEngine(async (engine) => {
          try {
            const saved = await engine.resumeChatCompletion(sessionId)
            const result = await engine.resumeChatCompletion(sessionId, {
              continueGeneration: true,
            })
            if (!('recoveredText' in saved) || !('recoveredText' in result))
              throw new Error('Expected native recovery results')
            return { saved, result }
          } finally {
            await engine.deleteResumableSession(sessionId)
          }
        }),
      sessionId,
    )
    expect(recovered.saved.recoveredText).toBe(checkpoint.text)
    expect(recovered.result.recoveryMode).toMatch(/^(kv|token_replay)$/)
    expect(recovered.result.recoveredText).toContain('30')

    const restored = await harness.create({ session, tools: [tag] })
    const reply = await restored.invoke(
      'What tag did get_tag return earlier? Use the existing result, without calling tools.',
      { limits: { turns: 3 } },
    )
    expect(restored.messages.slice(0, history.length).map((message) => message.toJSON())).toEqual(
      history,
    )
    expect(finalText(reply)).toContain(code)
    expect(calls).toBe(1)
  } finally {
    await harness.close()
  }
})
