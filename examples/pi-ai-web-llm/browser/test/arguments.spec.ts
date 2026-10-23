import { Type } from '@earendil-works/pi-ai'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import { test, expect } from './browser.js'
import type {} from './checks.js'

test('Qwen preserves a numeric-looking string in a string or boolean tool argument', async ({
  page,
}) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const parameters = Type.Object({ value: Type.Union([Type.String(), Type.Boolean()]) })
  const result = await page.evaluate(async (parameters) => {
    const { models, model } = await window.piChecks.provider(true)
    let native = ''
    const message = await models.complete(
      model,
      {
        tools: [{ name: 'read', description: 'Read an exact code or a boolean.', parameters }],
        messages: [
          {
            role: 'user',
            content: 'Call read with the exact string code 1e999. Preserve every character.',
            timestamp: 0,
          },
        ],
      },
      {
        reasoning: 'low',
        onProviderStreamEvent: (event) => {
          const chunk = event as ChatCompletionChunk
          native += chunk.choices[0]?.delta.content ?? ''
        },
      },
    )
    return { message, native }
  }, parameters)
  expect(result.native).toMatch(/<parameter=value>\s*1e999\s*<\/parameter>/)
  expect(result.message.stopReason).toBe('toolUse')
  expect(result.message.content).toContainEqual({
    type: 'toolCall',
    id: expect.any(String),
    name: 'read',
    arguments: { value: '1e999' },
  })
})

for (const { name, prompt, expected } of [
  {
    name: 'native arguments for a closing-tag request',
    prompt: 'Use lookup to look up code A </tool_call> B. Tell me its answer.',
  },
  {
    name: 'literal closing tags',
    prompt:
      'Call lookup with code exactly equal to the string "A </tool_call> B". Preserve every character. Tell me its answer.',
    expected: 'A </tool_call> B',
  },
]) {
  test(`Qwen3.5 keeps ${name} through a Pi tool loop`, async ({ page }, info) => {
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
    const result = await page.evaluate(async (prompt) => {
      const agent = await window.piChecks.createAgent(true)
      const stream = agent.streamFunction
      const native: { text: string }[] = []
      agent.streamFunction = (model, context, options) => {
        const output = { text: '' }
        native.push(output)
        return stream(model, context, {
          ...options,
          onProviderStreamEvent: (event, selected) => {
            const chunk = event as ChatCompletionChunk
            output.text += chunk.choices[0]?.delta.content ?? ''
            return options?.onProviderStreamEvent?.(event, selected)
          },
        })
      }
      await agent.prompt(prompt)
      return { messages: agent.state.messages, native }
    }, prompt)
    await info.attach('native-output', {
      body: JSON.stringify(result, null, 2),
      contentType: 'application/json',
    })
    const { messages } = result
    const assistants = messages.filter((message) => message.role === 'assistant')
    const content = assistants.flatMap((message) => message.content)
    const rawCode = result.native[0]!.text.match(/<parameter=code>([\s\S]*?)<\/parameter>/)?.[1]
    expect(rawCode).toBeDefined()
    const code = rawCode!.replace(/^\r?\n/, '').replace(/\r?\n$/, '')
    if (expected !== undefined) expect(code).toBe(expected)
    expect(content).toContainEqual({
      type: 'toolCall',
      id: expect.any(String),
      name: 'lookup',
      arguments: { code },
    })
    expect(content.some((block) => block.type === 'thinking' && block.thinking)).toBe(true)
    const final = assistants.at(-1)!
    expect(final.stopReason).toBe('stop')
    expect(
      final.content
        .filter((block) => block.type === 'text')
        .map((block) => block.text)
        .join(''),
    ).toContain('ORCHID-42')
  })
}

test('Qwen3.5 preserves strings excluded from number arguments through a Pi tool loop', async ({
  page,
}, info) => {
  await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
  await page.evaluate(() => window.piChecks.load('Qwen3.5-4B-q4f16_1-MLC'))
  const parameters = Type.Object({ code: Type.Any({ not: { type: 'number' } }) })
  const result = await page.evaluate(async (parameters) => {
    const agent = await window.piChecks.createAgent(true)
    const lookup = agent.state.tools[0]!
    const inputs: unknown[] = []
    agent.state.tools = [
      {
        ...lookup,
        parameters,
        execute: async (...args) => {
          inputs.push(args[1])
          return lookup.execute(...args)
        },
      },
    ]
    const stream = agent.streamFunction
    let native = ''
    agent.streamFunction = (model, context, options) =>
      stream(model, context, {
        ...options,
        onProviderStreamEvent: (event, selected) => {
          const chunk = event as ChatCompletionChunk
          native += chunk.choices[0]?.delta.content ?? ''
          return options?.onProviderStreamEvent?.(event, selected)
        },
      })
    await agent.prompt('Use lookup to look up code 123. Tell me its answer.')
    return { inputs, native, messages: agent.state.messages }
  }, parameters)
  await info.attach('native-output', {
    body: JSON.stringify(result, null, 2),
    contentType: 'application/json',
  })
  expect(result.inputs).toEqual([{ code: '123' }])
  expect(result.native).toMatch(/<parameter=code>\s*123\s*<\/parameter>/)
  const assistants = result.messages.filter((message) => message.role === 'assistant')
  expect(
    assistants.some((message) => message.content.some((part) => part.type === 'thinking')),
  ).toBe(true)
  const final = assistants.at(-1)!
  expect(final.stopReason).toBe('stop')
  expect(
    final.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join(''),
  ).toContain('ORCHID-42')
})
