import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createHarness, type HarnessAgentOptions } from '@strands-agents/harness'
import { expect, type Page } from '@playwright/test'
import type { AgentResult } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'
import { createMemory } from '../memory.js'
import { SummaryModel } from '../summary-model.js'
import { createRead } from '../read.js'
export { useTool } from '../tool-choice.js'

declare global {
  interface Window {
    gemmaPrompts: string[]
    gemmaModels: string[]
  }
}

/** Every test gets real defaults, isolated files, and one shared WebGPU engine. */
export async function openHarness(page: Page, contextWindow = 16384) {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  const dir = await mkdtemp(join(process.platform === 'win32' ? tmpdir() : '/tmp', 'gemma-'))
  const agents: Awaited<ReturnType<typeof createHarness>>[] = []
  await page.addInitScript(() => {
    window.gemmaPrompts = []
    window.gemmaModels = []
    const send = Worker.prototype.postMessage
    Worker.prototype.postMessage = function (this: Worker, message: unknown) {
      const request = message as {
        kind?: string
        content?: { modelId?: string[]; request?: { prompt?: string } }
      }
      if (request.kind === 'reload' && request.content?.modelId)
        window.gemmaModels = request.content.modelId
      if (request.kind === 'completionStreamInit' && request.content?.request?.prompt)
        window.gemmaPrompts.push(request.content.request.prompt)
      send.call(this, message)
    }
  })
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate(({ modelId, contextWindow }) => window.harness.open(modelId, contextWindow), {
    modelId,
    contextWindow,
  })
  expect(await page.evaluate(() => window.gemmaModels)).toEqual([modelId])
  const config = { modelId, contextWindowLimit: contextWindow, maxTokens: 4096, temperature: 0 }
  const model = new BrowserModel(page, {
    ...config,
    temperature: 1,
    topP: 0.95,
    request: { seed: 7, repetition_penalty: 1.05, extra_body: { enable_thinking: true } },
  })
  const auxModel = new SummaryModel(page, {
    ...config,
    maxTokens: 512,
  })
  const memory = createMemory(page, modelId, contextWindow, join(dir, 'memory'))
  return {
    dir,
    model,
    auxModel,
    async create(options: HarnessAgentOptions = {}) {
      const agent = await createHarness({
        model,
        auxModel,
        builtinTools: { read: false, web_fetch: { model: auxModel }, web_search: 'exa' },
        session: { dir: join(dir, 'sessions') },
        memory: { stores: [memory] },
        printer: false,
        ...options,
        tools: [createRead(), ...(options.tools ?? [])],
      })
      agents.push(agent)
      return agent
    },
    async close() {
      try {
        const results = await Promise.allSettled(agents.map((agent) => agent.shutdown()))
        for (const result of results) if (result.status === 'rejected') throw result.reason
      } finally {
        try {
          await page.evaluate(() => window.harness.close())
        } finally {
          await rm(dir, { recursive: true, force: true })
        }
      }
    },
  }
}

export function finalText(result: AgentResult) {
  return result.lastMessage.content
    .filter((block) => block.type === 'textBlock')
    .map((block) => block.text)
    .join('\n')
}
