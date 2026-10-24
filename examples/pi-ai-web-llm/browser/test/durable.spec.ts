import type { Page } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
import type { ConversationView, LiveState } from '@earendil-works/pi-durable'
import { test, expect } from './browser.js'

const url = 'http://127.0.0.1:4173/pi/durable/'
const defaultModel = 'Qwen3.5-4B-q4f16_1-MLC'

async function load(page: Page, modelId = defaultModel) {
  await page.evaluate((id) => {
    const select = document.querySelector<HTMLSelectElement>('#model')!
    if (![...select.options].some((option) => option.value === id)) select.add(new Option(id, id))
  }, modelId)
  await page.locator('#model').selectOption(modelId)
  await page.getByRole('button', { name: 'Load model and conversation', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('Ready', { timeout: 15 * 60 * 1000 })
}

async function fresh(page: Page, modelId = defaultModel) {
  await page.goto(url)
  await page.evaluate(
    (name) =>
      new Promise<void>((resolve, reject) => {
        const request = indexedDB.deleteDatabase(name)
        request.onsuccess = () => resolve()
        request.onerror = () => reject(request.error)
        request.onblocked = () => reject(new Error('The test database is still open.'))
      }),
    `pi-webllm-v2-${modelId}`,
  )
  await load(page, modelId)
}

const models = [
  'Qwen3.5-9B-q4f16_1-MLC',
  defaultModel,
  'Qwen3-4B-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
]

const defaultPrompt = 'Look up A7 and tell me the answer.'
const cases = [
  ...models.map((modelId) => ({
    modelId,
    prompt: defaultPrompt,
  })),
  {
    modelId: 'Qwen3.5-2B-q4f16_1-MLC',
    prompt: 'Call lookup with code A7 and report exactly the answer it returns.',
  },
]
const selected =
  process.env.MODEL_IDS?.split(',').map(
    (modelId) =>
      cases.find((item) => item.modelId === modelId) ?? { modelId, prompt: defaultPrompt },
  ) ?? cases

for (const { modelId, prompt } of selected) {
  test(`${modelId}: Durable tools and conversation survive a page reload`, async ({
    page,
  }, info) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await fresh(page, modelId)
    await page.locator('#message').fill(prompt)
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.locator('#status')).toHaveText('Done', { timeout: 120_000 })
    await expect(page.locator('#answer')).toContainText('ORCHID-42')
    const before: ConversationView = JSON.parse((await page.locator('#details').textContent())!)
    await writeFile(info.outputPath('conversation.json'), JSON.stringify(before, null, 2))
    expect(
      before.entries
        .flatMap((entry) => entry.model ?? [])
        .some(
          (message) =>
            message.role === 'assistant' &&
            message.content.some(
              (part) =>
                part.type === 'toolCall' && part.name === 'lookup' && part.arguments.code === 'A7',
            ),
        ),
    ).toBe(true)
    expect(before.entries.some((entry) => entry.kind === 'pi.tool-result')).toBe(true)
    expect(JSON.stringify(before.docs['pi.usage'])).toMatch(/"totalTokens":[1-9]/)
    await load(page, modelId)
    const reopened = JSON.parse((await page.locator('#details').textContent())!)
    expect(reopened.entries).toEqual(before.entries)
    expect(reopened.docs['pi.usage']).toEqual(before.docs['pi.usage'])
    await page.reload()
    await load(page, modelId)
    const restored = JSON.parse((await page.locator('#details').textContent())!)
    expect(restored.conversation.id).toBe(before.conversation.id)
    expect(restored.entries).toEqual(before.entries)
    expect(restored.docs['pi.usage']).toEqual(before.docs['pi.usage'])
    await expect(page.locator('#answer')).toContainText('ORCHID-42')
    await page
      .locator('#message')
      .fill('Repeat the answer you already looked up. Do not call tools.')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.locator('#status')).toHaveText('Done', { timeout: 120_000 })
    await expect(page.locator('#answer')).toContainText('ORCHID-42')
    await info.attach('conversation', {
      body: (await page.locator('#details').textContent())!,
      contentType: 'application/json',
    })
    expect(errors).toEqual([])
  })
}

test('Durable can abort a running turn and use the same model again', async ({ page }) => {
  await fresh(page)
  await page
    .locator('#message')
    .fill('Count from 1 to 1000. Use digits. Do not explain or use tools.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  // Observe and cancel together, before the running turn can finish.
  await page.waitForFunction(
    () => {
      const view = JSON.parse(document.querySelector('#details')!.textContent!) as ConversationView
      const live = view.docs['pi.live'] as LiveState | undefined
      const content = live?.generation?.message?.content
      if (
        !content?.some(
          (part) =>
            (part.type === 'text' && part.text) || (part.type === 'thinking' && part.thinking),
        )
      )
        return false
      const cancel = document.querySelector<HTMLButtonElement>('#cancel')!
      if (cancel.disabled) return false
      cancel.click()
      return true
    },
    undefined,
    { timeout: 120_000 },
  )
  await expect(page.locator('#status')).toHaveText('Stopped', { timeout: 30_000 })
  await page.locator('#message').fill('Look up A7 and tell me the answer.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('Done', { timeout: 120_000 })
  await expect(page.locator('#answer')).toContainText('ORCHID-42')
})

test('Durable resumes an unfinished model turn after the page is destroyed', async ({ page }) => {
  await fresh(page)
  await page
    .locator('#message')
    .fill('Count from 1 to 100. Use digits. Do not explain or use tools.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#answer')).toContainText('1', { timeout: 120_000 })
  await expect(page.locator('#status')).toHaveText('Running')
  await page.reload()
  await load(page)
  await expect(page.locator('#answer')).toContainText('100')
  const restored = JSON.parse((await page.locator('#details').textContent())!)
  expect(restored.entries.some((entry: { kind: string }) => entry.kind === 'pi.assistant')).toBe(
    true,
  )
  expect(restored.docs['pi.live'].run).toBeUndefined()
})

test('a second Durable page cannot write the same saved conversation', async ({
  page,
  context,
}) => {
  await fresh(page)
  const other = await context.newPage()
  try {
    await other.goto(url)
    await other.getByRole('button', { name: 'Load model and conversation', exact: true }).click()
    await expect(other.locator('#status')).toContainText('already open in another tab', {
      timeout: 120_000,
    })
    await expect(other.getByRole('button', { name: 'Send', exact: true })).toBeDisabled()
  } finally {
    await other.close()
  }
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('#status')).toHaveText('Done', { timeout: 120_000 })
  await expect(page.locator('#answer')).toContainText('ORCHID-42')
})
