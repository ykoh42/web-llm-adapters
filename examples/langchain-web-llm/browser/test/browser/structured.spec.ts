import { z } from 'zod'
import { createAgent, toolStrategy } from 'langchain'
import { tool } from '@langchain/core/tools'
import { AIMessage } from '@langchain/core/messages'
import { writeFile } from 'node:fs/promises'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../../shared/sampling.js'

test('Qwen thinking executes a tool before returning an agent response schema', async ({
  page,
}, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const secret = 'secret-' + crypto.randomUUID().slice(0, 8)
  const inputs: unknown[] = []
  const lookup = tool(
    (input) => {
      inputs.push(input)
      return { secret }
    },
    {
      name: 'lookup',
      description: 'Look up a secret by its code.',
      schema: z.object({ code: z.string() }),
    },
  )
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const agent = createAgent({
    model,
    tools: [lookup],
    responseFormat: toolStrategy(z.object({ secret: z.string() })),
  })
  const result = await agent.invoke({
    messages: [
      {
        role: 'user',
        content: 'Use lookup with code A7, then return its secret in the required format.',
      },
    ],
  })
  await writeFile(
    info.outputPath('structured-agent.json'),
    JSON.stringify({ inputs, result }, null, 2),
  )
  expect(inputs).toEqual([{ code: 'A7' }])
  expect(result.structuredResponse).toEqual({ secret })
  expect(
    result.messages.filter(AIMessage.isInstance).flatMap((message) => message.contentBlocks),
  ).toContainEqual(expect.objectContaining({ type: 'reasoning' }))
})

for (const modelId of ['gemma-4-E2B-it-q4f16_1-MLC', 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC']) {
  test(`${modelId} schema extraction uses its native function format`, async ({ page }) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: {
        ...(modelId.startsWith('gemma-') ? sampling(modelId, true) : { temperature: 0 }),
        max_tokens: 4096,
        seed: 7,
      },
    })
    const result = await model
      .withStructuredOutput(z.object({ answer: z.number() }), { includeRaw: true })
      .invoke('What is six times seven?')
    expect(result.parsed).toEqual({ answer: 42 })
    expect(result.raw).toMatchObject({ tool_calls: [{ name: 'extract', args: { answer: 42 } }] })
  })
}

for (const method of ['jsonMode', 'jsonSchema']) {
  test(`${method} keeps native JSON output when call options change sampling`, async ({
    page,
  }, testInfo) => {
    const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
    await page.evaluate((id) => window.checks.load(id), modelId)
    if (method === 'jsonMode') {
      const failure = await page.evaluate(async (modelId) => {
        try {
          const stream = await window.checks.engine().chat.completions.create({
            model: modelId,
            messages: [
              { role: 'user', content: 'Return JSON with answer set to six times seven.' },
            ],
            response_format: { type: 'json_object' },
            max_tokens: 128,
            stream: true,
          })
          for await (const chunk of stream) void chunk
          return undefined
        } catch (error) {
          return String(error)
        }
      }, modelId)
      expect(failure).toContain('GrammarMatcherInitError:')
      expect(failure).toContain('Cannot pass non-string to std::string')
      await testInfo.attach('native-json-mode-error', {
        body: JSON.stringify(failure),
        contentType: 'application/json',
      })
    }
    const direct = await page.evaluate(async (method) => {
      return window.checks
        .model()
        .withStructuredOutput(
          {
            type: 'object',
            properties: { answer: { type: 'number' } },
            required: ['answer'],
          },
          { method },
        )
        .invoke('Return JSON with answer set to six times seven.', {
          request: { temperature: 0, max_tokens: 128 },
        })
    }, method)
    expect(direct).toEqual({ answer: 42 })

    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, false), seed: 7 },
    })
    const result = await model
      .withStructuredOutput(z.object({ answer: z.number() }), { method, includeRaw: true })
      .invoke('Return JSON with answer set to six times seven.', {
        request: { temperature: 0, max_tokens: 128 },
      })

    expect(result.parsed).toEqual({ answer: 42 })
    expect(JSON.parse(result.raw.text)).toEqual(result.parsed)
  })
}

test('bare JSON mode preserves nested objects and root arrays', async ({ page }) => {
  const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const expected = [{ data: [1, true, null, { text: 'oak' }] }, [1, true, null, { text: 'oak' }]]

  for (const value of expected) {
    const result = await page.evaluate(async (value) => {
      const message = await window.checks
        .model()
        .invoke('Return only this exact JSON: ' + JSON.stringify(value), {
          request: { response_format: { type: 'json_object' }, temperature: 0, max_tokens: 128 },
        })
      return JSON.parse(message.text)
    }, value)
    expect(result).toEqual(value)
  }
})
