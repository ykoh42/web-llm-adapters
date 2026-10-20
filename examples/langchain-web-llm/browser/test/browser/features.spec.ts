import { z } from 'zod'
import type { JSONSchema7 } from 'json-schema'
import { createDeepAgent } from 'deepagents'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { InMemoryStore } from '@langchain/langgraph'
import { AIMessage, AIMessageChunk, HumanMessage } from '@langchain/core/messages'
import { BrowserChatWebLLM, BrowserWebLLMEmbeddings } from 'langchain-web-llm'
import { writeFile } from 'node:fs/promises'
import type { WebLLMChunk } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../../shared/sampling.js'
import { sharedSchema } from '../../../../shared/schema.js'

test('uses a Web Worker by default', async ({ page }) => {
  await expect(page.getByLabel('Web Worker')).toBeChecked()
})

function lookupSchema(code: JSONSchema7, key = 'code') {
  return { type: 'object', properties: { [key]: code }, required: [key] } satisfies JSONSchema7
}

const schemaCases = [
  { name: 'enum', schema: lookupSchema({ enum: ['123', '456'] }) },
  { name: 'const', schema: lookupSchema({ const: '123' }) },
  {
    name: 'spaced argument',
    schema: lookupSchema({ type: 'string' }, 'access code'),
    key: 'access code',
  },
  {
    name: 'typed conditions',
    schema: lookupSchema({
      type: 'string',
      oneOf: [{ pattern: '^1' }, { pattern: '^4' }],
    }),
  },
  { name: 'property oneOf', schema: lookupSchema({ oneOf: [{ const: '123' }, { const: '456' }] }) },
  { name: 'shared references', schema: sharedSchema() },
  {
    name: 'property pattern',
    schema: {
      type: 'object',
      patternProperties: { '^code$': { type: 'string' } },
      additionalProperties: false,
      required: ['code'],
    },
  },
  {
    name: 'array reference',
    schema: lookupSchema({
      $ref: '#/properties/code/definitions/value/anyOf/0',
      definitions: { value: { anyOf: [{ type: 'string' }] } },
    }),
  },
  {
    name: 'anyOf',
    schema: {
      type: 'object',
      anyOf: [lookupSchema({ const: '123' }), lookupSchema({ const: '456' })],
    },
  },
  {
    name: 'oneOf',
    schema: {
      type: 'object',
      oneOf: [lookupSchema({ const: '123' }), lookupSchema({ const: '456' })],
    },
  },
] satisfies { name: string; schema: JSONSchema7; key?: string }[]

for (const { name, schema, key = 'code' } of schemaCases) {
  test(`Qwen3.5 keeps ${name} strings through a LangChain tool loop`, async ({ page }, info) => {
    const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
    await page.evaluate((id) => window.checks.load(id), modelId)
    const code = 'code-' + crypto.randomUUID().slice(0, 8)
    const inputs: unknown[] = []
    const raw: WebLLMChunk[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => raw.push(chunk),
    })
    const lookup = tool(
      (input) => {
        inputs.push(input)
        return `The secret is ${code}.`
      },
      {
        name: 'lookup',
        description: 'Look up the secret for a code.',
        schema,
      },
    )
    const agent = createAgent({ model, tools: [lookup] })
    const result = await agent
      .invoke({
        messages: [
          { role: 'user', content: 'Use lookup to look up code 123. Tell me its secret.' },
        ],
      })
      .catch(async (error: unknown) => {
        await writeFile(
          info.outputPath('schema.json'),
          JSON.stringify({ schema, inputs, raw, error: String(error) }, null, 2),
        )
        throw error
      })
    await writeFile(
      info.outputPath('schema.json'),
      JSON.stringify({ schema, inputs, raw, result }, null, 2),
    )
    expect(inputs).toEqual([{ [key]: '123' }])
    expect(result.messages.at(-1)?.text).toContain(code)
    expect(
      result.messages.filter(AIMessage.isInstance).flatMap((message) => message.contentBlocks),
    ).toContainEqual(expect.objectContaining({ type: 'reasoning' }))
  })
}

test('Qwen3.5 preserves encoded references before LangChain tool validation', async ({
  page,
}, info) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const schema = {
    type: 'object',
    properties: {
      code: {
        $ref: '#/properties/code/definitions/value%7E1text',
        definitions: { 'value/text': { type: 'string' } },
      },
    },
    required: ['code'],
  } satisfies JSONSchema7
  const inputs: unknown[] = []
  const lookup = tool((input) => inputs.push(input), {
    name: 'lookup',
    description: 'Look up the secret for a code.',
    schema,
  })
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const response = await model
    .bindTools([lookup])
    .invoke('Use lookup to look up code 123. Tell me its secret.')
  expect(response.tool_calls).toMatchObject([{ name: 'lookup', args: { code: '123' } }])
  await writeFile(info.outputPath('schema.json'), JSON.stringify({ schema, response }, null, 2))
  // This failure also occurs without WebLLM or an adapter.
  await expect(lookup.invoke({ code: '123' })).rejects.toThrow('Unresolved $ref')
  expect(inputs).toEqual([])
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId}: callbacks and agent streams preserve reasoning, tools and the final answer`, async ({
    page,
  }) => {
    await page.evaluate((id) => window.checks.load(id, true), modelId)
    const observed = await page.evaluate(async () => {
      const chunks: { reasoning: boolean; tools: unknown; finish: unknown }[] = []
      await window.checks
        .model(true)
        .bindTools([
          {
            type: 'function',
            function: {
              name: 'lookup',
              description: 'Look up the answer for a code.',
              parameters: {
                type: 'object',
                properties: { code: { type: 'string' } },
                required: ['code'],
              },
            },
          },
        ])
        .invoke('Look up A7 and tell me the answer.', {
          callbacks: [
            {
              handleLLMNewToken(...args) {
                const chunk = args[5]?.chunk
                if (!chunk || !('message' in chunk)) return
                chunks.push({
                  reasoning: chunk.message.contentBlocks.some(
                    (block) => block.type === 'reasoning',
                  ),
                  tools: 'tool_calls' in chunk.message ? chunk.message.tool_calls : [],
                  finish: chunk.message.response_metadata.finish_reason,
                })
              },
            },
          ],
        })
      return chunks
    })
    expect(observed.some((chunk) => chunk.reasoning)).toBe(true)
    expect(observed.flatMap((chunk) => chunk.tools)).toContainEqual(
      expect.objectContaining({ name: 'lookup', args: { code: 'A7' } }),
    )
    expect(observed.at(-1)?.finish).toBe('tool_calls')
    let firstStopped = false
    let liveReasoning = false
    const model = new BrowserChatWebLLM(page, {
      modelId,
      profile: await page.evaluate(() => window.checks.model(true).profile),
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk: (chunk) => {
        if (chunk.choices[0]?.finish_reason) firstStopped = true
      },
    })
    const codes: string[] = []
    const lookup = tool(
      ({ code }) => {
        codes.push(code)
        return `The answer for ${code} is 42.`
      },
      {
        name: 'lookup',
        description: 'Look up the answer for a code.',
        schema: z.object({ code: z.string() }),
      },
    )
    const agent = createAgent({ model, tools: [lookup] })
    const chunks: AIMessageChunk[] = []
    for await (const [message] of await agent.stream(
      {
        messages: [{ role: 'user', content: 'Look up A7 and tell me the answer.' }],
      },
      { streamMode: 'messages' },
    )) {
      if (AIMessageChunk.isInstance(message)) {
        chunks.push(message)
        if (!firstStopped && message.contentBlocks.some((block) => block.type === 'reasoning'))
          liveReasoning = true
      }
    }
    const reasoning = chunks
      .flatMap((message) => message.contentBlocks)
      .filter((block) => block.type === 'reasoning')
    expect(reasoning.length).toBeGreaterThan(0)
    expect(liveReasoning).toBe(true)
    expect(chunks.flatMap((message) => message.tool_calls ?? [])).toContainEqual(
      expect.objectContaining({ name: 'lookup', args: { code: 'A7' } }),
    )
    expect(codes).toEqual(['A7'])
    expect(chunks.map((message) => message.text).join('')).toContain('42')
  })
}

test('real WebLLM embeddings rank LangGraph semantic memory', async ({ page }) => {
  const modelId = 'snowflake-arctic-embed-s-q0f32-MLC-b4'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const embeddings = new BrowserWebLLMEmbeddings(page, { modelId })
  const vector = await embeddings.embedQuery('A flower')
  expect(vector).toHaveLength(384)
  expect(vector.every(Number.isFinite)).toBe(true)
  const store = new InMemoryStore({ index: { embeddings, dims: 384 } })
  await store.put(['notes'], 'flower', { text: 'An orchid is a flowering plant.' })
  await store.put(['notes'], 'car', { text: 'A car is a vehicle with wheels and an engine.' })
  expect((await store.search(['notes'], { query: 'a flower', limit: 1 }))[0]?.key).toBe('flower')
})

test('real vision model accepts LangChain standard image blocks', async ({ page }) => {
  const modelId = 'Phi-3.5-vision-instruct-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const data = await page.evaluate(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 224
    canvas.height = 224
    const context = canvas.getContext('2d')!
    context.fillStyle = 'red'
    context.fillRect(0, 0, 224, 224)
    return canvas.toDataURL('image/png').split(',')[1]!
  })
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { max_tokens: 256, temperature: 0 },
  })
  const response = await model.invoke([
    new HumanMessage({
      content: [
        { type: 'text', text: 'What color is this image? Answer with the color.' },
        { type: 'image', source_type: 'base64', data, mime_type: 'image/png' },
      ],
    }),
  ])
  expect(response.text.toLowerCase()).toContain('red')
  const color = await model.withStructuredOutput(z.object({ color: z.string() })).invoke([
    new HumanMessage({
      content: [
        { type: 'text', text: 'Return the color of this image.' },
        {
          type: 'image',
          data: new Uint8Array(Buffer.from(data, 'base64')),
          mimeType: 'image/png',
        },
      ],
    }),
  ])
  expect(color.color.toLowerCase()).toContain('red')
})

test('real engine serializes concurrent inference, cancels generation and recovers', async ({
  page,
}) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const responses = await Promise.all([
    model.invoke('What is 1 + 1?'),
    model.invoke('What is 2 + 2?'),
  ])
  expect(responses[0]?.text).toContain('2')
  expect(responses[1]?.text).toContain('4')
  const cancel = new AbortController()
  const stream = await model.stream('Explain the history of mathematics in detail.', {
    signal: cancel.signal,
  })
  const first = await stream.next()
  expect(first.done).toBe(false)
  cancel.abort()
  await expect(stream.next()).rejects.toBeDefined()
  expect((await model.invoke('What is 3 + 3?')).text).toContain('6')
})

test('default Deep Agents subagents share the WebLLM engine and file state', async ({ page }) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: await page.evaluate(() => window.checks.model().profile),
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const agent = createDeepAgent({ model })
  const tools: string[] = []
  const result = await agent.invoke(
    {
      messages: [
        {
          role: 'user',
          content:
            'Use the task tool with a general-purpose subagent to read /answer.txt. Then tell me the secret code from its report.',
        },
      ],
      files: {
        '/answer.txt': {
          content: ['The secret code is ORCHID-42.'],
          created_at: '2026-10-19',
          modified_at: '2026-10-19',
        },
      },
    },
    {
      callbacks: [
        {
          handleToolStart(...args) {
            if (args[6]) tools.push(args[6])
          },
        },
      ],
    },
  )
  expect(tools).toContain('task')
  expect(tools).toContain('read_file')
  expect(result.messages.at(-1)?.text).toContain('ORCHID-42')
})

test('the Deep Agents example writes and reads a file with native generation defaults', async ({
  page,
}) => {
  const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: await page.evaluate(() => window.checks.model(true).profile),
    request: { extra_body: { enable_thinking: true } },
  })
  const agent = createDeepAgent({ model })
  const tools: string[] = []
  const result = await agent.invoke(
    {
      messages: [
        {
          role: 'user',
          content: 'Write a short greeting to /greeting.txt, then read it.',
        },
      ],
    },
    {
      callbacks: [
        {
          handleToolStart(...args) {
            if (args[6]) tools.push(args[6])
          },
        },
      ],
    },
  )
  expect(tools).toContain('write_file')
  expect(tools).toContain('read_file')
  expect(result.files?.['/greeting.txt']?.content.length).toBeGreaterThan(0)
  expect(result.messages.at(-1)?.type).toBe('ai')
  expect(result.messages.at(-1)?.text.length).toBeGreaterThan(0)
})

test('Deep Agents receives Qwen tool arguments unchanged', async ({ page }, info) => {
  const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  await page.evaluate((id) => window.checks.load(id), modelId)
  const raw: WebLLMChunk[] = []
  const model = new BrowserChatWebLLM(page, {
    modelId,
    profile: await page.evaluate(() => window.checks.model(false).profile),
    request: { ...sampling(modelId, false), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => raw.push(chunk),
  })
  const agent = createDeepAgent({ model })
  const tools: string[] = []
  const result = await agent.invoke(
    {
      messages: [
        {
          role: 'user',
          content: 'Write this exact text to /tags.txt, then read it: Closing tag: </tool_call>',
        },
      ],
    },
    {
      callbacks: [
        {
          handleToolStart(...args) {
            if (args[6]) tools.push(args[6])
          },
        },
      ],
    },
  )
  await writeFile(info.outputPath('inference.json'), JSON.stringify({ raw, result }))
  expect(tools).toContain('write_file')
  expect(tools).toContain('read_file')
  const text = raw
    .map((chunk) => {
      const choice = chunk.choices[0]
      return choice && 'delta' in choice ? (choice.delta.content ?? '') : ''
    })
    .join('')
  const native = [...text.matchAll(/<tool_call>\s*({[\s\S]*?})\s*<\/tool_call>/g)].map((match) =>
    JSON.parse(match[1]!),
  )
  const calls = result.messages
    .filter(AIMessage.isInstance)
    .flatMap((message) =>
      (message.tool_calls ?? []).map((call) => ({ name: call.name, arguments: call.args })),
    )
  // A model may omit requested text; the adapter must preserve what it actually generated.
  expect(calls).toEqual(native)
  const written = calls.find((call) => call.name === 'write_file')
  expect(result.files?.['/tags.txt']?.content).toEqual(written?.arguments.content)
  expect(result.messages.at(-1)?.type).toBe('ai')
  expect(result.messages.at(-1)?.text.length).toBeGreaterThan(0)
})

for (const modelId of ['Qwen3.5-4B-q4f16_1-MLC', 'gemma-4-E2B-it-q4f16_1-MLC']) {
  test(`${modelId}: a WebLLM Worker runs default Deep Agents and handles cancellation`, async ({
    page,
  }) => {
    await page.evaluate((id) => window.checks.load(id, true), modelId)
    const model = new BrowserChatWebLLM(page, {
      modelId,
      profile: await page.evaluate(() => window.checks.model(true).profile),
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    })
    const agent = createDeepAgent({ model })
    const result = await agent.invoke({
      messages: [
        { role: 'user', content: 'Read /answer.txt and tell me the secret code exactly.' },
      ],
      files: {
        '/answer.txt': {
          content: ['The secret code is ORCHID-42.'],
          created_at: '2026-10-19',
          modified_at: '2026-10-19',
        },
      },
    })
    expect(result.messages.at(-1)?.text).toContain('ORCHID-42')
    const cancel = new AbortController()
    const stream = await model.stream('Explain the history of mathematics in detail.', {
      signal: cancel.signal,
    })
    expect((await stream.next()).done).toBe(false)
    cancel.abort()
    await expect(stream.next()).rejects.toBeDefined()
    expect((await model.invoke('What is 3 + 3?')).text).toContain('6')
  })
}
