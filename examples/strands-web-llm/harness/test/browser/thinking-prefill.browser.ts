import { Agent, FunctionTool } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import type { ChatOptions } from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

test('Strands uses prefilled Qwen thinking through a real tool loop', async ({
  page,
  request,
}, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  const response = await request.get(
    `https://huggingface.co/mlc-ai/${modelId}/resolve/main/mlc-chat-config.json`,
  )
  expect(response.ok()).toBe(true)
  const config: ChatOptions = await response.json()
  expect(config.conv_template).toBeDefined()
  const template = { ...config.conv_template!, role_empty_sep: '\n<think>\n' }
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  try {
    await page.evaluate(
      ({ modelId, template }) =>
        window.webllmBridge.withEngine((engine) =>
          engine.reload(modelId, { context_window_size: 32768, conv_template: template }),
        ),
      { modelId, template },
    )
    const settings = sampling(modelId, true)
    const raw: string[] = []
    const model = new BrowserModel(page, {
      modelId,
      thinkingPrefilled: true,
      ...settings,
      maxTokens: 4096,
      request: { ...settings.request, seed: 7 },
      onChunk(chunk) {
        const choice = chunk.choices[0]
        if (choice && 'delta' in choice) {
          if (raw.length === 0) raw.push('')
          raw[raw.length - 1] += choice.delta.content ?? ''
          if (choice.finish_reason) raw.push('')
        }
      },
    })
    const inputs: unknown[] = []
    const lookup = new FunctionTool({
      name: 'lookup',
      description: 'Look up the answer for a code.',
      inputSchema: {
        type: 'object',
        properties: { code: { type: 'string' } },
        required: ['code'],
      },
      callback(input) {
        inputs.push(input)
        return 'The answer for A7 is 42.'
      },
    })
    const agent = new Agent({ model, tools: [lookup], printer: false })
    const result = await agent.invoke('Use lookup to look up A7 and tell me its answer.')
    const reasoning = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'reasoningBlock')
      .map((block) => block.text ?? '')
    await info.attach('prefilled-tool-loop', {
      body: JSON.stringify({ raw, reasoning, messages: agent.messages }),
      contentType: 'application/json',
    })
    expect(inputs).toEqual([{ code: 'A7' }])
    expect(String(result)).toContain('42')
    expect(reasoning.some((text) => text.trim())).toBe(true)
    expect(raw.filter(Boolean)).toHaveLength(2)
    for (const [index, text] of raw.filter(Boolean).entries()) {
      const end = text.indexOf('</think>')
      expect(end).toBeGreaterThan(0)
      expect(reasoning[index]).toBe(text.slice(0, end))
    }
    expect(result.stopReason).toBe('endTurn')
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})
