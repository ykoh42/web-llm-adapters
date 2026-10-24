import { BrowserChatWebLLM } from 'langchain-web-llm'
import type { ChatOptions } from '@mlc-ai/web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of [
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
]) {
  test(`${modelId} reads thinking prefilled by the loaded template`, async ({
    page,
    request,
  }, info) => {
    const response = await request.get(
      `https://huggingface.co/mlc-ai/${modelId}/resolve/main/mlc-chat-config.json`,
    )
    expect(response.ok()).toBe(true)
    const config: ChatOptions = await response.json()
    expect(config.conv_template).toBeDefined()
    const template = { ...config.conv_template!, role_empty_sep: '\n<think>\n' }
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(
      ({ modelId, template }) =>
        window.webllmLangChainBridge!.withEngine((engine) =>
          engine.reload(modelId, { context_window_size: 32768, conv_template: template }),
        ),
      { modelId, template },
    )
    let raw = ''
    const model = new BrowserChatWebLLM(page, {
      modelId,
      thinkingPrefilled: true,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk(chunk) {
        const choice = chunk.choices[0]
        if (choice && 'delta' in choice) raw += choice.delta.content ?? ''
      },
    })
    const result = await model.invoke('What is 17 * 23?')
    const native = await page.evaluate(
      ({ modelId, request }) =>
        window.webllmLangChainBridge!.withEngine(async (engine) => {
          await engine.resetChat()
          let text = ''
          for await (const chunk of await engine.chat.completions.create({
            ...request,
            model: modelId,
            messages: [{ role: 'user', content: 'What is 17 * 23?' }],
            stream: true,
            stream_options: { include_usage: true },
            n: 1,
          }))
            text += chunk.choices[0]?.delta.content ?? ''
          return text
        }),
      { modelId, request: model.invocationParams({}) },
    )
    const reasoning = result.contentBlocks
      .filter((part) => part.type === 'reasoning')
      .map((part) => part.reasoning)
      .join('')
    await info.attach('prefilled-thinking', {
      body: JSON.stringify({ raw, native, reasoning, text: result.text }),
      contentType: 'application/json',
    })
    expect(raw).toBe(native)
    const end = native.indexOf('</think>')
    expect(end).toBeGreaterThan(0)
    expect(reasoning.trim()).not.toBe('')
    expect(reasoning).toBe(native.slice(0, end))
    expect(result.text).toBe(native.slice(end + '</think>'.length))
    expect(result.text).toContain('391')
    expect(result.response_metadata.finish_reason).toBe('stop')
  })
}
