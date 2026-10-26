import { modelCheck } from '../../../../../scripts/release/support.js'
import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { Agent, BeforeModelCallEvent } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { recordNative, replayNative } from '../../fixtures/native.js'

test('Qwen2.5 literal thinking tags reach Strands unchanged', async ({ page }) => {
  const modelId = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((id) => window.harness.open(id), modelId)
  let native = ''
  const model = new BrowserModel(page, {
    modelId,
    temperature: 0,
    maxTokens: 256,
    onChunk: (chunk) => {
      const choice = chunk.choices[0]
      if (choice && 'delta' in choice) native += choice.delta.content ?? ''
    },
  })
  try {
    const agent = new Agent({ model, printer: false })
    const result = await agent.invoke('Repeat this text exactly:\n<think>literal</think>')

    expect(native).toContain('<think>')
    expect(result.lastMessage.content).toEqual([
      expect.objectContaining({ type: 'textBlock', text: native }),
    ])
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})

const cases = [
  {
    modelId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    // Native generation omits a character here; preserve the original failing input.
    sample: { file: 'q25-76746.txt', code: 'code-4b62881b' },
    exactCopy: false,
    release: true,
  },
  ...[
    'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  ].map((modelId) => ({ modelId, sample: undefined, exactCopy: true, release: true })),
  {
    modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    sample: { file: 'q25-70021.txt', code: 'code-77b175b5' },
    exactCopy: true,
    release: false,
  },
]

for (const { modelId, sample, exactCopy, release } of cases) {
  const title = `${modelId} calls Harness read and answers from its result`
  test(
    title + (sample ? ' with recorded tool history' : ''),
    release ? modelCheck(modelId, 'tools') : {},
    async ({ page }, info) => {
      const directory = process.platform === 'win32' ? tmpdir() : '/tmp'
      const path = join(directory, sample?.file ?? `q25-${process.pid}.txt`)
      const code = sample?.code ?? 'code-' + crypto.randomUUID().slice(0, 8)
      const chunks: WebLLMChunk[][] = []
      let agent: Awaited<ReturnType<typeof createHarness>> | undefined
      let created = false

      try {
        await writeFile(path, `The secret code is ${code}.`, { flag: 'wx' })
        created = true
        await page.goto('http://127.0.0.1:4173/harness.html')
        await page.waitForFunction(() => Boolean(window.harness))
        await page.evaluate((id) => window.harness.open(id), modelId)
        await page.evaluate(() =>
          window.webllmBridge.withEngine((engine) => {
            window.qwen25Requests = []
            const endpoint = engine.chat.completions
            endpoint.create = new Proxy(endpoint.create, {
              apply(target, self, args) {
                window.qwen25Requests.push(structuredClone(args[0]))
                return Reflect.apply(target, self, args)
              },
            })
          }),
        )

        agent = await createHarness({
          model: new BrowserModel(page, {
            modelId,
            maxTokens: 1024,
            temperature: 0,
            onChunk: (chunk) => chunks.at(-1)!.push(chunk),
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
          chunks.push([])
        })

        const result = await agent.invoke(
          `Use the read tool once with offset 1 to read ${path}. After it returns, answer with the secret code. Do not call the read tool again.`,
          { limits: { turns: 3 } },
        )

        const requests = await page.evaluate(() => window.qwen25Requests)
        const native = await replayNative(page, requests)
        const text = result.lastMessage.content
          .filter((block) => block.type === 'textBlock')
          .map((block) => block.text)
          .join('')
        await recordNative(info, 'native-comparison', {
          code,
          path,
          requests,
          adapted: chunks,
          native,
          text,
          messages: agent.messages.map((message) => message.toJSON()),
        })
        expect(modelCalls).toBe(2)
        expect(requests).toHaveLength(2)
        expect(chunks.map(rawText)).toEqual(native.map(rawText))
        expect(result.stopReason).toBe('endTurn')
        expect(text.trim()).not.toBe('')
        expect(text).toBe(rawText(native.at(-1)!))
        expect(
          native
            .at(-1)!
            .flatMap((chunk) => chunk.choices.map((choice) => choice.finish_reason))
            .filter(Boolean),
        ).toEqual(['stop'])
        if (exactCopy) expect(text).toContain(code)
        const blocks = agent.messages.flatMap((message) => message.content)
        const calls = blocks.filter((block) => block.type === 'toolUseBlock')
        expect(calls).toHaveLength(1)
        expect(calls[0]).toMatchObject({ name: 'read', input: { path } })
        const results = blocks.filter((block) => block.type === 'toolResultBlock')
        expect(results).toHaveLength(1)
        expect(results[0]?.status).toBe('success')
        expect(JSON.stringify(results)).toContain(code)
        const response = requests[1]!.messages.at(-1)
        expect(response?.role).toBe('user')
        expect(response?.content).toContain('<tool_response>')
        expect(response?.content).toContain(`The secret code is ${code}.`)
      } finally {
        try {
          await agent?.shutdown()
          await page.evaluate(() => window.harness?.close())
        } finally {
          if (created) await rm(path, { force: true })
          await info.attach('native-chunks', {
            body: JSON.stringify(chunks),
            contentType: 'application/json',
          })
        }
      }
    },
  )
}

function rawText(chunks: readonly WebLLMChunk[]) {
  return chunks
    .flatMap((chunk) =>
      chunk.choices.map((choice) =>
        'delta' in choice ? (choice.delta.content ?? '') : choice.text,
      ),
    )
    .join('')
}

declare global {
  interface Window {
    qwen25Requests: ChatCompletionRequestStreaming[]
  }
}
