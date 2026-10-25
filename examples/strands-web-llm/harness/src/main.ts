import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { createHarness } from '@strands-agents/harness'
import { chromium } from '@playwright/test'
import { BrowserModel } from 'strands-web-llm'
import { useTool } from './tool-choice.ts'
import { createEmbeddingMemory, createMemory, embeddingModelId } from './memory.ts'
import { createRead } from './tools/read.ts'
import { chatOptions } from '../../chat-options.ts'
import { sampling } from '../../sampling.ts'

const { values, positionals } = parseArgs({
  options: {
    model: { type: 'string' },
    think: { type: 'boolean' },
    tool: { type: 'string' },
    'semantic-memory': { type: 'boolean' },
  },
  allowPositionals: true,
})
const thinking = values.think ?? false
const semanticMemory = values['semantic-memory'] ?? false
const modelId = values.model ?? 'gemma-4-E2B-it-q4f16_1-MLC'
const contextWindow = 16384
const browser = await chromium.launchPersistentContext(
  fileURLToPath(new URL('../../../../.cache/chrome', import.meta.url)),
  {
    channel: 'chrome',
    headless: false,
  },
)
try {
  const page = browser.pages()[0] ?? (await browser.newPage())
  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate(({ modelId, contextWindow }) => window.harness.open(modelId, contextWindow), {
    modelId,
    contextWindow,
  })
  if (semanticMemory) {
    await page.evaluate(
      ({ modelId, embeddingModelId, options }) =>
        window.webllmBridge.withEngine((engine) =>
          engine.reload([modelId, embeddingModelId], [options, {}]),
        ),
      {
        modelId,
        embeddingModelId,
        options: { ...chatOptions(modelId), context_window_size: contextWindow },
      },
    )
  }
  const config = {
    modelId,
    contextWindowLimit: contextWindow,
    thinkingPrefilled: modelId.startsWith('Qwen3.5-'),
  }
  const auxModel = new BrowserModel(page, {
    ...config,
    maxTokens: 512,
    ...sampling(modelId, false),
  })
  const agent = await createHarness({
    model: new BrowserModel(page, {
      ...config,
      maxTokens: 4096,
      ...sampling(modelId, thinking),
    }),
    auxModel,
    memory: {
      stores: [
        semanticMemory
          ? createEmbeddingMemory(page, modelId, contextWindow, '.agent/semantic-memory')
          : createMemory(page, modelId, contextWindow),
      ],
    },
    instructions: 'Answer briefly.',
    builtinTools: { read: false, web_fetch: { model: auxModel }, web_search: 'exa' },
    tools: [createRead()],
    printer: false,
  })
  if (values.tool) useTool(agent, values.tool)
  try {
    const result = await agent.invoke(positionals[0] ?? 'Say hello.')
    console.log(
      result.lastMessage.content
        .filter((block) => block.type === 'textBlock')
        .map((block) => block.text)
        .join('\n'),
    )
  } finally {
    await agent.shutdown()
  }
} finally {
  await browser.close()
}
