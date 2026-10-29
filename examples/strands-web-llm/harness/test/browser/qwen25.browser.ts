import { modelCheck } from '../../../../../scripts/release/support.js'
import { rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { Agent, BeforeModelCallEvent, BeforeToolCallEvent } from '@strands-agents/sdk'
import { expect, test } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { recordNative } from '../../fixtures/native.js'

const qwen25Model = 'Qwen2.5-3B-Instruct-q4f16_1-MLC'

test(
  'Qwen2.5 preserves native text with thinking disabled',
  modelCheck(qwen25Model),
  async ({ page }) => {
    const modelId = qwen25Model
    await page.goto('http://127.0.0.1:4173/harness.html')
    await page.waitForFunction(() => Boolean(window.harness))
    await page.evaluate((id) => window.harness.open(id), modelId)
    let native = ''
    const model = new BrowserModel(page, {
      modelId,
      temperature: 0,
      maxTokens: 1024,
      request: { extra_body: { enable_thinking: false } },
      onChunk: (chunk) => {
        const choice = chunk.choices[0]
        if (choice && 'delta' in choice) native += choice.delta.content ?? ''
      },
    })
    try {
      const agent = new Agent({ model, printer: false })
      const result = await agent.invoke('Repeat this text exactly:\n<think>literal</think>')

      const prefix = '<think>\n\n</think>\n\n'
      expect(native.startsWith(prefix)).toBe(true)
      expect(result.lastMessage.content).toEqual([
        expect.objectContaining({ type: 'textBlock', text: native.slice(prefix.length) }),
      ])
    } finally {
      await page.evaluate(() => window.harness.close())
    }
  },
)

const cases = [
  {
    modelId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    // Native generation omits a character here; preserve the original failing input.
    sample: { file: 'q25-76746.txt', code: 'code-4b62881b' },
    release: true,
  },
  ...[
    'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  ].map((modelId) => ({ modelId, sample: undefined, release: true })),
  {
    modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    sample: { file: 'q25-70021.txt', code: 'code-77b175b5' },
    release: false,
  },
  {
    modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    sample: { file: 'q25-62169.txt', code: 'code-38adf3b0' },
    release: true,
  },
].flatMap((item) =>
  (item.modelId === 'Qwen2.5-7B-Instruct-q4f16_1-MLC' ? [false, true] : [false]).map(
    (constrainedToolCalls) => ({ ...item, constrainedToolCalls }),
  ),
)

for (const { modelId, sample, release, constrainedToolCalls } of cases) {
  const protocol = modelId === 'Qwen2.5-7B-Instruct-q4f16_1-MLC' && !constrainedToolCalls
  const title = protocol
    ? `${modelId} preserves native tool output through Harness`
    : `${modelId} calls Harness read and preserves the native answer`
  const feature = constrainedToolCalls ? 'constrained-tools' : 'tools'
  test(
    title +
      (sample ? ` with recorded input ${sample.file}` : '') +
      (constrainedToolCalls ? ' with constraints' : ''),
    release ? modelCheck(modelId, protocol ? 'tool-protocol' : feature) : {},
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
            request: { constrainedToolCalls },
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
        let toolCalls = 0
        agent.addHook(BeforeModelCallEvent, () => {
          modelCalls++
          chunks.push([])
        })
        agent.addHook(BeforeToolCallEvent, () => {
          toolCalls++
        })

        const outcome = await agent
          .invoke(
            `Use the read tool once with offset 1 to read ${path}. After it returns, answer with the secret code. Do not call the read tool again.`,
            { limits: { turns: 3 } },
          )
          .then(
            (result) => ({ result }),
            (error: unknown) => ({ error }),
          )

        const requests = await page.evaluate(() => window.qwen25Requests)
        await recordNative(info, 'native-comparison', {
          code,
          path,
          requests,
          native: chunks,
          error: 'error' in outcome ? String(outcome.error) : undefined,
          messages: agent.messages.map((message) => message.toJSON()),
        })
        if ('error' in outcome) {
          if (!protocol) throw outcome.error
          expect(outcome.error).toMatchObject({ message: 'Qwen returned invalid tool JSON.' })
          expect(modelCalls).toBe(1)
          expect(toolCalls).toBe(0)
          expect(requests).toHaveLength(1)
          expect(requests[0]).not.toHaveProperty('response_format')
          const raw = rawText(chunks[0]!)
          expect(raw.match(/<tool_call>/g)).toHaveLength(1)
          expect(raw.match(/<\/tool_call>/g)).toHaveLength(1)
          const source = raw.match(/<tool_call>([\s\S]*)<\/tool_call>/)?.[1]
          expect(source).toBeDefined()
          expect(() => JSON.parse(source!)).toThrow(SyntaxError)
          return
        }
        const { result } = outcome
        const text = result.lastMessage.content
          .filter((block) => block.type === 'textBlock')
          .map((block) => block.text)
          .join('')
        if (constrainedToolCalls) expect(requests[0]?.response_format?.type).toBe('structural_tag')
        else expect(requests[0]).not.toHaveProperty('response_format')
        expect(modelCalls).toBe(2)
        expect(requests).toHaveLength(2)
        expect(result.stopReason).toBe('endTurn')
        expect(text.trim()).not.toBe('')
        // Keep the native answer even when the model copies the code incorrectly.
        expect(text).toBe(rawText(chunks.at(-1)!))
        expect(
          chunks
            .at(-1)!
            .flatMap((chunk) => chunk.choices.map((choice) => choice.finish_reason))
            .filter(Boolean),
        ).toEqual(['stop'])
        const blocks = agent.messages.flatMap((message) => message.content)
        const calls = blocks.filter((block) => block.type === 'toolUseBlock')
        expect(calls).toHaveLength(1)
        expect(calls[0]).toMatchObject({ name: 'read', input: { path } })
        const source = rawText(chunks[0]!).match(/<tool_call>([\s\S]*?)<\/tool_call>/)?.[1]
        expect(source).toBeDefined()
        expect(JSON.parse(source!)).toMatchObject({
          name: calls[0]!.name,
          arguments: calls[0]!.input,
        })
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
