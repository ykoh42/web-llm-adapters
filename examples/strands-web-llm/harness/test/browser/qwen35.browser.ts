import { modelCheck } from '../../../../../scripts/release/support.js'
import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { Agent, BeforeModelCallEvent, FunctionTool, type JSONSchema } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel } from 'strands-web-llm'
import { sampling } from '../../../sampling.js'
import { sharedSchema } from '../../../../../test/support/schema.js'

function lookupSchema(code: JSONSchema, key = 'code') {
  return { type: 'object', properties: { [key]: code }, required: [key] } satisfies JSONSchema
}

const schemaCases = [
  {
    name: 'literal closing tags',
    schema: lookupSchema({ type: 'string' }),
    value: 'A </tool_call> B',
  },
  { name: 'enum', schema: lookupSchema({ enum: ['123', '456'] }) },
  { name: 'excluded number type', schema: lookupSchema({ not: { type: 'number' } }) },
  {
    name: 'object exclusion',
    schema: { ...lookupSchema({ type: 'string' }), not: lookupSchema({ type: 'number' }) },
  },
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
    name: 'encoded reference',
    schema: lookupSchema({
      $ref: '#/properties/code/definitions/value%7E1text',
      definitions: { 'value/text': { type: 'string' } },
    }),
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
] satisfies { name: string; schema: JSONSchema; key?: string; value?: string }[]

for (const { name, schema, key = 'code', value = '123' } of schemaCases) {
  test(`Qwen3.5 keeps ${name} strings through a Strands tool loop`, async ({ page }) => {
    const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    try {
      const code = 'code-' + crypto.randomUUID().slice(0, 8)
      const inputs: unknown[] = []
      const settings = sampling(modelId, true)
      const model = new BrowserModel(page, {
        modelId,
        thinkingPrefilled: true,
        ...settings,
        maxTokens: 4096,
        request: { ...settings.request, seed: 7 },
      })
      const lookup = new FunctionTool({
        name: 'lookup',
        description: 'Look up the secret for a code.',
        inputSchema: schema,
        callback: (input) => {
          inputs.push(input)
          return `The secret is ${code}.`
        },
      })
      const agent = new Agent({ model, tools: [lookup], printer: false })
      const result = await agent.invoke(`Use lookup to look up code ${value}. Tell me its secret.`)
      expect(inputs).toEqual([{ [key]: value }])
      expect(String(result)).toContain(code)
      expect(
        agent.messages
          .flatMap((message) => message.content)
          .some((block) => block.type === 'reasoningBlock' && block.text),
      ).toBe(true)
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  })
}

// Keep the filenames from both native API regressions.
for (const [modelId, fileName] of [
  ['Qwen3.5-0.8B-q4f16_1-MLC', 'q35-diagnosis.txt'],
  ['Qwen3.5-0.8B-q4f16_1-MLC', 'q35-40625.txt'],
  ['Qwen3.5-2B-q4f16_1-MLC', `q35-${process.pid}.txt`],
  ['Qwen3.5-4B-q4f16_1-MLC', `q35-${process.pid}.txt`],
  ['Qwen3.5-9B-q4f16_1-MLC', `q35-${process.pid}.txt`],
] as const) {
  const note = modelId === 'Qwen3.5-0.8B-q4f16_1-MLC' ? ` (${fileName})` : ''
  test(
    `${modelId} thinking calls Harness read and answers${note}`,
    fileName === 'q35-40625.txt' ? {} : modelCheck(modelId, 'tools', 'thinking'),
    async ({ page }) => {
      const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
      const path = join(directory, fileName)
      const code = 'code-' + crypto.randomUUID().slice(0, 8)
      let agent: Awaited<ReturnType<typeof createHarness>> | undefined

      await writeFile(path, `The secret code is ${code}.`, { flag: 'wx' })
      try {
        await page.goto('http://127.0.0.1:4173/harness.html')
        await page.waitForFunction(() => Boolean(window.harness))
        await page.evaluate((id) => window.harness.open(id), modelId)

        agent = await createHarness({
          model: new BrowserModel(page, {
            modelId,
            thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
            temperature: 0,
            request: { extra_body: { enable_thinking: true } },
          }),
          builtinTools: ['read'],
          builtinPlugins: [],
          session: false,
          skills: false,
          memory: false,
          contextManager: false,
          caching: false,
          backgroundTasks: false,
          printer: false,
        })
        let modelCalls = 0
        agent.addHook(BeforeModelCallEvent, () => {
          modelCalls++
        })

        const result = await agent.invoke(
          `Use the read tool to read ${path}. What secret code does it contain?`,
          { limits: { turns: 3 } },
        )

        expect(modelCalls).toBe(2)
        expect(result.stopReason).toBe('endTurn')
        expect(result.lastMessage.content).toContainEqual(
          expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(code) }),
        )
        expect(
          agent.messages
            .flatMap((message) => message.content)
            .some((block) => block.type === 'reasoningBlock' && block.text),
        ).toBe(true)
        const toolCalls = agent.messages
          .flatMap((message) => message.content)
          .filter((block) => block.type === 'toolUseBlock')
        expect(toolCalls).toHaveLength(1)
        expect(toolCalls[0]).toMatchObject({ name: 'read', input: { path } })
        const toolResults = agent.messages
          .flatMap((message) => message.content)
          .filter((block) => block.type === 'toolResultBlock')
        expect(toolResults.length).toBeGreaterThan(0)
        expect(JSON.stringify(toolResults)).toContain(code)
      } finally {
        try {
          await agent?.shutdown()
          await page.evaluate(() => window.harness?.close())
        } finally {
          await rm(path, { force: true })
        }
      }
    },
  )
}
