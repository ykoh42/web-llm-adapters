import { writeFile } from 'node:fs/promises'
import {
  Agent,
  tool as defineTool,
  JsonBlock,
  Message,
  TextBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'
import { modelCheck } from '../../../../../scripts/release/support.js'

for (const modelId of ['Qwen3-1.7B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
  test(
    `${modelId} constrains required tools and preserves thinking through the final answer`,
    modelCheck(modelId, 'tools', 'thinking'),
    async ({ page }, info) => {
      const receipt = 'CODE-' + crypto.randomUUID().slice(0, 8)
      const inputs: unknown[] = []
      const inputSchema = z
        .object({ city: z.string(), unit: z.enum(['celsius', 'fahrenheit']) })
        .strict()
      const tool = defineTool({
        name: 'lookup',
        description: 'Process the supplied fields and return a verification code.',
        inputSchema,
        callback: (input) => {
          inputs.push(input)
          return { verification_code: receipt }
        },
      })
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id, 8192), modelId)
      const settings = sampling(modelId, true)
      const chunks: WebLLMChunk[][] = []
      const model = new BrowserModel(page, {
        modelId,
        ...settings,
        contextWindowLimit: 8192,
        maxTokens: 4096,
        request: { ...settings.request, constrainedToolCalls: true, seed: 17 },
        onChunk: (chunk) => chunks.at(-1)!.push(chunk),
      })
      try {
        for (const toolChoice of [{ any: {} }, { tool: { name: 'lookup' } }]) {
          const messages = [
            new Message({
              role: 'user',
              content: [
                new TextBlock(
                  'Call lookup for Seoul with Celsius. Then report the returned verification code.',
                ),
              ],
            }),
          ]
          chunks.push([])
          const first = await read(
            model.streamAggregated(messages, { toolSpecs: [tool.toolSpec], toolChoice }),
          )
          const calls = first.filter((block) => block.type === 'toolUseBlock')
          expect(calls).toHaveLength(1)
          const call = calls[0]!
          expect(call.name).toBe('lookup')
          expect(call.input).toEqual({ city: 'Seoul', unit: 'celsius' })
          const raw = chunks
            .at(-1)!
            .map((chunk) => {
              const choice = chunk.choices[0]
              return choice && 'delta' in choice ? (choice.delta.content ?? '') : ''
            })
            .join('')
          const reasoning = first
            .filter((block) => block.type === 'reasoningBlock')
            .map((block) => block.text ?? '')
            .join('')
          expect(reasoning).toBe(
            [...raw.matchAll(/<think>([\s\S]*?)<\/think>/g)].map((match) => match[1]).join(''),
          )
          const result = await tool.invoke(inputSchema.parse(call.input))
          messages.push(
            new Message({ role: 'assistant', content: first }),
            new Message({
              role: 'user',
              content: [
                new ToolResultBlock({
                  toolUseId: call.toolUseId,
                  status: 'success',
                  content: [new JsonBlock({ json: result })],
                }),
              ],
            }),
          )
          chunks.push([])
          const last = await read(model.streamAggregated(messages, { toolSpecs: [tool.toolSpec] }))
          expect(last.some((block) => block.type === 'toolUseBlock')).toBe(false)
          expect(
            last
              .filter((block) => block.type === 'textBlock')
              .map((block) => block.text)
              .join(''),
          ).toContain(receipt)
        }
        expect(inputs).toHaveLength(2)
        const path = info.outputPath('constrained-tools.json')
        await writeFile(path, JSON.stringify({ modelId, inputs, chunks }, null, 2))
        await info.attach('constrained-tools', { path, contentType: 'application/json' })
      } finally {
        await page.evaluate(() => window.harness.close())
      }
    },
  )
}

for (const modelId of [
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen3-1.7B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
]) {
  const thinking = !modelId.startsWith('Qwen2.5-')
  test(
    `${modelId} constrains an Agent tool loop and preserves native output`,
    thinking ? modelCheck(modelId, 'tools', 'thinking') : modelCheck(modelId, 'tools'),
    async ({ page }, info) => {
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id, 16384), modelId)
      try {
        for (const constrainedToolCalls of [false, true]) {
          const receipt = 'CODE-' + crypto.randomUUID().slice(0, 8)
          const inputs: unknown[] = []
          const chunks: WebLLMChunk[] = []
          const settings = sampling(modelId, thinking)
          const model = new BrowserModel(page, {
            modelId,
            ...settings,
            contextWindowLimit: 16384,
            maxTokens: 4096,
            request: { ...settings.request, constrainedToolCalls, seed: 7 },
            onChunk: (chunk) => chunks.push(chunk),
          })
          const lookup = defineTool({
            name: 'lookup',
            description: 'Look up a code.',
            inputSchema: z.strictObject({ code: z.string(), enabled: z.boolean() }),
            callback: (input) => {
              inputs.push(input)
              return receipt
            },
          })
          const agent = new Agent({ model, tools: [lookup], printer: false })
          await agent.invoke(
            'Call lookup with code A7 and enabled false. After its result, reply with the returned code.',
          )
          await info.attach('agent-' + constrainedToolCalls, {
            body: JSON.stringify({
              constrainedToolCalls,
              receipt,
              inputs,
              messages: agent.messages,
              chunks,
            }),
            contentType: 'application/json',
          })
          expect(inputs).toEqual([{ code: 'A7', enabled: false }])
          const finalText = agent.messages
            .at(-1)!
            .content.filter((block) => block.type === 'textBlock')
            .map((block) => block.text)
            .join('')
          expect(finalText.trim()).not.toBe('')
          const reasoning = agent.messages
            .flatMap((message) =>
              message.content
                .filter((block) => block.type === 'reasoningBlock')
                .map((block) => block.text ?? ''),
            )
            .join('')
          const raw = chunks
            .map((chunk) => {
              const choice = chunk.choices[0]
              return choice && 'delta' in choice ? (choice.delta.content ?? '') : ''
            })
            .join('')
          const text = agent.messages
            .flatMap((message) =>
              message.role === 'assistant'
                ? message.content
                    .filter((block) => block.type === 'textBlock')
                    .map((block) => block.text)
                : [],
            )
            .join('')
          const turns = ['']
          for (const chunk of chunks) {
            const choice = chunk.choices[0]
            if (!choice || !('delta' in choice)) continue
            turns[turns.length - 1] += choice.delta.content ?? ''
            if (choice.finish_reason) turns.push('')
          }
          // Strands removes whitespace-only TextBlocks when it aggregates a message.
          const nativeText = turns
            .map((turn) =>
              thinking
                ? turn.replace(/<think>[\s\S]*?<\/think>|<tool_call>[\s\S]*?<\/tool_call>/g, '')
                : turn.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, ''),
            )
            .filter((turn) => turn.trim())
            .join('')
          expect(text).toBe(nativeText)
          expect(reasoning).toBe(
            thinking
              ? [...raw.matchAll(/<think>([\s\S]*?)<\/think>/g)].map((match) => match[1]).join('')
              : '',
          )
        }
      } finally {
        await page.evaluate(() => window.harness.close())
      }
    },
  )
}

async function read(stream: ReturnType<BrowserModel['streamAggregated']>) {
  for (;;) {
    const next = await stream.next()
    if (next.done) return next.value.message.content
  }
}
