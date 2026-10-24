import { modelCheck } from '../../../../../scripts/release/support.js'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness } from '@strands-agents/harness'
import { test, expect } from '../../../fixtures/browser.js'
import { BrowserModel, type WebLLMChunk } from 'strands-web-llm'

declare global {
  interface Window {
    gemmaPrompts: string[]
  }
}

test(
  'Gemma thinks, calls Harness read, answers, and strips old thoughts on the next turn',
  modelCheck('gemma-4-E2B-it-q4f16_1-MLC', 'tools', 'thinking'),
  async ({ page }) => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    const directory = await mkdtemp(join(tmpdir(), 'gemma-thinking-'))
    const path = join(directory, 'note.txt')
    const code = 'oak-' + crypto.randomUUID().slice(0, 8)
    try {
      await writeFile(path, `The verification code is ${code}.`)
      await page.addInitScript(() => {
        window.gemmaPrompts = []
        const send = Worker.prototype.postMessage
        // WebLLM sends its request with one argument. Observe it without changing the model.
        Worker.prototype.postMessage = function (this: Worker, message: unknown) {
          const request = message as { kind?: string; content?: { request?: { prompt?: string } } }
          if (request.kind === 'completionStreamInit' && request.content?.request?.prompt)
            window.gemmaPrompts.push(request.content.request.prompt)
          send.call(this, message)
        }
      })
      await page.goto('http://127.0.0.1:4173/harness.html')
      await page.waitForFunction(() => Boolean(window.harness))
      await page.evaluate((id) => window.harness.open(id), modelId)
      const chunks: WebLLMChunk[] = []
      const model = new BrowserModel(page, {
        modelId,
        maxTokens: 1024,
        temperature: 0,
        onChunk: (chunk) => chunks.push(chunk),
        request: { extra_body: { enable_thinking: true } },
      })
      const agent = await createHarness({
        model,
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
      const order: string[] = []
      let answer = ''
      for await (const update of agent.stream(
        `Use read to read ${path}. What is the verification code?`,
        {
          limits: { turns: 3 },
        },
      )) {
        if (update.type === 'toolResultEvent') order.push('tool')
        if (
          update.type !== 'modelStreamUpdateEvent' ||
          update.event.type !== 'modelContentBlockDeltaEvent'
        )
          continue
        const delta = update.event.delta
        if (delta.type === 'reasoningContentDelta' && delta.text) order.push('thinking')
        if (delta.type === 'textDelta') {
          order.push('answer')
          answer += delta.text
        }
      }
      const thoughts = agent.messages
        .flatMap((message) => message.content)
        .filter((block) => block.type === 'reasoningBlock')
        .filter((block) => block.text)
      expect(thoughts.length).toBeGreaterThan(0)
      expect(order.indexOf('thinking')).toBeLessThan(order.indexOf('tool'))
      expect(order.lastIndexOf('answer')).toBeGreaterThan(order.indexOf('tool'))
      expect(answer).toContain(code)
      expect(answer).not.toMatch(/<\|channel>|<channel\|>|<\|tool_call>/)
      expect(chunks.every((chunk) => chunk.object === 'text_completion')).toBe(true)
      const raw = chunks
        .flatMap((chunk) =>
          chunk.object === 'text_completion' ? chunk.choices.map((choice) => choice.text) : [],
        )
        .join('')
      expect(raw).toContain('<|tool_call>')
      expect(raw).toContain(code)
      expect(chunks.some((chunk) => chunk.usage?.total_tokens)).toBe(true)
      const prompts = await page.evaluate(() => window.gemmaPrompts)
      expect(prompts).toHaveLength(2)
      expect(prompts[0]).toContain('<|think|>')
      expect(prompts[1]).toContain('<|channel>thought\n' + thoughts[0]!.text + '\n<channel|>')
      expect(prompts[1]).toContain('<|tool_response>response:read')
      expect(prompts[1]?.endsWith('<|channel>thought\n')).toBe(true)
      const next = await agent.invoke('From our conversation, what was the verification code?', {
        limits: { turns: 3 },
      })
      expect(next.lastMessage.content).toContainEqual(
        expect.objectContaining({
          type: 'textBlock',
          text: expect.stringContaining(code),
        }),
      )
      const later = await page.evaluate(() => window.gemmaPrompts[2])
      expect(later).not.toContain('<|channel>thought\n')
      expect(later).not.toContain(thoughts[0]!.text)
      expect(agent.messages.flatMap((message) => message.content)).toContain(thoughts[0])
      const controller = new AbortController()
      try {
        for await (const update of agent.stream('What is 47 multiplied by 19?', {
          cancelSignal: controller.signal,
        })) {
          if (
            update.type === 'modelStreamUpdateEvent' &&
            update.event.type === 'modelContentBlockDeltaEvent' &&
            update.event.delta.type === 'reasoningContentDelta'
          )
            controller.abort()
        }
      } catch (error) {
        if (!controller.signal.aborted) throw error
      }
      expect(controller.signal.aborted).toBe(true)
      const recovered = await agent.invoke('From our conversation, repeat the verification code.', {
        limits: { turns: 3 },
      })
      expect(recovered.lastMessage.content).toContainEqual(
        expect.objectContaining({
          type: 'textBlock',
          text: expect.stringContaining(code),
        }),
      )
      model.updateConfig({
        request: { tool_choice: 'none', extra_body: { enable_thinking: false } },
      })
      chunks.length = 0
      const disabled = await agent.invoke('Repeat the verification code without using tools.', {
        limits: { turns: 3 },
      })
      expect(disabled.lastMessage.content).toContainEqual(
        expect.objectContaining({ type: 'textBlock', text: expect.stringContaining(code) }),
      )
      expect(chunks.every((chunk) => chunk.object === 'text_completion')).toBe(true)
      const history = await page.evaluate(() => window.gemmaPrompts.at(-1))
      expect(history).toContain('<|tool_response>response:read')
      expect(history).not.toContain('declaration:')
      expect(history).not.toContain('<|think|>')
      console.log('Gemma thinking loop verified; final answer:', answer)
    } finally {
      try {
        await page.evaluate(() => window.harness?.close())
      } finally {
        await rm(directory, { recursive: true, force: true })
      }
    }
  },
)
