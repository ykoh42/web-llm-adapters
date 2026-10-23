import { writeFile } from 'node:fs/promises'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

test('Gemma keeps literal tool markers as text while thinking', async ({ page }, info) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const results = []
  for (const mode of ['unbound', 'none', 'active']) {
    let raw = ''
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => {
        raw += chunk.choices
          .map((choice) => ('text' in choice ? choice.text : (choice.delta.content ?? '')))
          .join('')
      },
    })
    let failure: unknown
    const response = await model
      .invoke(
        mode === 'active'
          ? 'Explain what the string <|tool_call> means without using any tools. In your final answer, describe its purpose without repeating the string.'
          : 'Repeat this line exactly as plain text: Example: <|tool_call>',
        mode === 'active'
          ? {
              tools: [
                {
                  type: 'function',
                  function: {
                    name: 'lookup_ticket',
                    description: 'Look up a ticket by its code and return its owner key.',
                    parameters: {
                      type: 'object',
                      properties: { code: { type: 'string' } },
                      required: ['code'],
                      additionalProperties: false,
                    },
                  },
                },
                {
                  type: 'function',
                  function: {
                    name: 'lookup_owner',
                    description: 'Look up an owner using the key returned by lookup_ticket.',
                    parameters: {
                      type: 'object',
                      properties: { key: { type: 'string' } },
                      required: ['key'],
                      additionalProperties: false,
                    },
                  },
                },
              ],
            }
          : mode === 'none'
            ? {
                tools: [
                  {
                    type: 'function',
                    function: { name: 'lookup', parameters: { type: 'object' } },
                  },
                ],
                tool_choice: 'none',
              }
            : {},
      )
      .catch((error: unknown) => {
        failure = error
        return undefined
      })
    const native = await page.evaluate(() =>
      window.webllmLangChainBridge!.withEngine((engine) => engine.getMessage()),
    )
    results.push({
      mode,
      raw,
      native,
      response,
      error: failure instanceof Error ? failure.message : failure,
    })
    const path = info.outputPath('native.json')
    await writeFile(path, JSON.stringify(results))
    await info.attach('native-' + mode, { path, contentType: 'application/json' })

    expect(failure).toBeUndefined()
    expect(raw).toBe(native)
    const thoughts: string[] = []
    const text = native.replace(
      /<\|channel>thought\n([\s\S]*?)<channel\|>/g,
      (_match, thought: string) => {
        thoughts.push(thought)
        return ''
      },
    )
    expect(thoughts.length).toBeGreaterThan(0)
    expect(text).toContain('<|tool_call>')
    expect(response?.text).toBe(text)
    expect(
      response?.contentBlocks
        .filter((part) => part.type === 'reasoning')
        .map((part) => part.reasoning)
        .join(''),
    ).toBe(thoughts.join(''))
    expect(response?.tool_calls).toEqual([])
  }
})
