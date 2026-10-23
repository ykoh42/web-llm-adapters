import {
  AIMessage,
  type AIMessageChunk,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { BaseCallbackHandler } from '@langchain/core/callbacks/base'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { writeFile } from 'node:fs/promises'
import { createAgent } from 'langchain'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import { z } from 'zod'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'

test.beforeEach(async ({ page }) => {
  await page.evaluate((id) => window.checks.load(id), modelId)
  await page.evaluate(() => {
    const engine = window.checks.engine()
    window.xmlRequests = []
    engine.chat.completions.create = new Proxy(engine.chat.completions.create, {
      apply(target, self, args) {
        window.xmlRequests.push(structuredClone(args[0]))
        return Reflect.apply(target, self, args)
      },
    })
  })
})

test('Qwen keeps a literal function closer inside a string argument', async ({ page }, info) => {
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  let response: AIMessageChunk<StandardMessageStructure> | undefined
  let failure: unknown
  try {
    response = await model
      .bindTools([
        {
          type: 'function',
          function: {
            name: 'save_text',
            description: 'Save text exactly as provided.',
            parameters: {
              type: 'object',
              properties: { text: { type: 'string' } },
              required: ['text'],
            },
          },
        },
      ])
      .invoke('Call save_text with text exactly: A </function> B')
  } catch (error) {
    failure = error
  }
  const native = await page.evaluate(async () => {
    const engine = window.checks.engine()
    await engine.resetChat()
    let text = ''
    let finish: string | null = null
    for await (const chunk of await engine.chat.completions.create(window.xmlRequests[0]!)) {
      text += chunk.choices[0]?.delta.content ?? ''
      finish = chunk.choices[0]?.finish_reason ?? finish
    }
    return { text, finish }
  })
  await info.attach('native', { body: JSON.stringify(native), contentType: 'application/json' })
  expect(native.finish).toBe('stop')
  expect(/<parameter=text>\s*([\s\S]*?)\s*<\/parameter>/.exec(native.text)?.[1]).toBe(
    'A </function> B',
  )
  expect(failure).toBeUndefined()
  expect(response?.tool_calls).toMatchObject([
    { name: 'save_text', args: { text: 'A </function> B' } },
  ])
})

test('an agent replays literal function tags after executing its tool', async ({ page }) => {
  const text = 'A </function> B'
  const saved: string[] = []
  const save = tool(
    ({ text }) => {
      saved.push(text)
      return text
    },
    {
      name: 'save_text',
      description: 'Save text exactly as provided.',
      schema: z.object({ text: z.string() }),
    },
  )
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
  })
  const agent = createAgent({ model, tools: [save] })
  const result = await agent.invoke({
    messages: [
      {
        role: 'user',
        content:
          'Save exactly the text between backticks using save_text: `A </function> B`. ' +
          'After the tool returns, reply with the saved text.',
      },
    ],
  })
  const responses = result.messages.filter(AIMessage.isInstance)
  expect(saved).toEqual([text])
  expect(responses).toHaveLength(2)
  expect(responses[0]?.tool_calls).toMatchObject([{ name: 'save_text', args: { text } }])
  expect(responses[1]?.text).toContain(text)
  expect(responses[1]?.response_metadata.finish_reason).toBe('stop')
  expect(
    responses.some((message) => message.contentBlocks.some((block) => block.type === 'reasoning')),
  ).toBe(true)
  const requests = await page.evaluate(() => window.xmlRequests)
  expect(requests).toHaveLength(2)
  const replay = requests[1]?.messages.find((message) => message.role === 'assistant')?.content
  expect(replay).toContain(`<parameter=text>\n${text}\n</parameter>`)
  const reply = requests[1]?.messages.at(-1)?.content
  expect(reply).toContain(`<tool_response>\n${text}\n</tool_response>`)
})

test('an agent preserves every native JSON tool call containing XML parameter tags', async ({
  page,
}, info) => {
  const input = {
    payload: { text: 'A <parameter=x> B </parameter> C' },
    items: ['</function>', '</tool_call>'],
  }
  const saved: (typeof input)[] = []
  const save = tool(
    (args) => {
      saved.push(args)
      return 'Saved'
    },
    {
      name: 'save_data',
      description: 'Save structured data exactly as provided.',
      schema: z.object({
        payload: z.object({ text: z.string() }),
        items: z.array(z.string()),
      }),
    },
  )
  const chunks: WebLLMChunk[] = []
  const nativeMessages: string[] = []
  const capture = BaseCallbackHandler.fromMethods({
    handleLLMEnd: async () => {
      nativeMessages.push(
        await page.evaluate(
          (id) => window.webllmLangChainBridge!.withEngine((engine) => engine.getMessage(id)),
          modelId,
        ),
      )
    },
  })
  capture.awaitHandlers = true
  capture.raiseError = true
  const model = new BrowserChatWebLLM(page, {
    modelId,
    request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
    onChunk: (chunk) => chunks.push(chunk),
    callbacks: [capture],
  })
  const agent = createAgent({ model, tools: [save] })
  let response
  let failure: unknown
  try {
    response = await agent.invoke({
      messages: [
        {
          role: 'user',
          content:
            'Call save_data with these exact arguments: ' +
            JSON.stringify(input) +
            '. After the tool returns, reply Saved.',
        },
      ],
    })
  } catch (error) {
    failure = error
  }
  const requests = await page.evaluate(() => window.xmlRequests)
  const native = await page.evaluate(async (requests) => {
    const engine = window.checks.engine()
    const outputs = []
    for (const request of requests) {
      await engine.resetChat()
      let text = ''
      let finish: string | null = null
      for await (const chunk of await engine.chat.completions.create(request)) {
        text += chunk.choices[0]?.delta.content ?? ''
        finish = chunk.choices[0]?.finish_reason ?? finish
      }
      outputs.push({ text, finish })
    }
    return outputs
  }, requests)
  const evidence = info.outputPath('json-parameter-tags.json')
  await writeFile(
    evidence,
    JSON.stringify({
      input,
      saved,
      requests,
      chunks,
      nativeMessages,
      native,
      error: String(failure),
    }),
  )
  await info.attach('json-parameter-tags', { path: evidence, contentType: 'application/json' })

  const actual = new Map<string, string>()
  for (const chunk of chunks) {
    const text = chunk.choices
      .map((choice) => ('delta' in choice ? (choice.delta.content ?? '') : choice.text))
      .join('')
    actual.set(chunk.id, (actual.get(chunk.id) ?? '') + text)
  }
  expect([...actual.values()]).toEqual(nativeMessages)
  expect(native.every((output) => output.finish === 'stop')).toBe(true)
  expect(failure).toBeUndefined()
  // Check native repeats too; the adapter must not invent or remove calls.
  const calls = native.reduce(
    (count, output) => count + [...output.text.matchAll(/<function=save_data>/g)].length,
    0,
  )
  expect(calls).toBeGreaterThan(0)
  expect(saved).toHaveLength(calls)
  for (const args of saved) expect(args).toEqual(input)
  for (const request of requests) {
    expect(request.extra_body?.enable_thinking).toBe(true)
    expect(request.max_tokens).toBe(4096)
    expect(request.seed).toBe(7)
  }
  const last = response?.messages.at(-1)
  const final = nativeMessages.at(-1)!
  const thoughts = /<think>([\s\S]*?)<\/think>/g
  expect(last?.text).toBe(final.replace(thoughts, ''))
  expect(
    last?.contentBlocks
      .filter((block) => block.type === 'reasoning')
      .map((block) => block.reasoning)
      .join(''),
  ).toBe([...final.matchAll(thoughts)].map((match) => match[1]).join(''))
  expect(last?.text).toContain('Saved')
  expect(last?.response_metadata.finish_reason).toBe('stop')
})

declare global {
  interface Window {
    xmlRequests: ChatCompletionRequestStreaming[]
  }
}
