import type { Page } from '@playwright/test'
import { test, expect } from './browser.js'

const url = 'http://127.0.0.1:4173/pi/durable/'
const defaultModel = 'Qwen3.5-4B-q4f16_1-MLC'

async function load(page: Page, modelId = defaultModel) {
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
    `pi-webllm-${modelId}`,
  )
  await load(page, modelId)
}

for (const modelId of [
  defaultModel,
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
]) {
  test(`${modelId}: Durable tools and conversation survive a page reload`, async ({
    page,
  }, info) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await fresh(page, modelId)
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.locator('#status')).toHaveText('Done', { timeout: 120_000 })
    await expect(page.locator('#answer')).toContainText('ORCHID-42')
    const before = JSON.parse((await page.locator('#details').textContent())!)
    expect(before.entries.some((entry: { kind: string }) => entry.kind === 'pi.tool-result')).toBe(
      true,
    )
    expect(JSON.stringify(before.docs['pi.usage'])).toMatch(/"totalTokens":[1-9]/)
    await page.reload()
    await load(page, modelId)
    const restored = JSON.parse((await page.locator('#details').textContent())!)
    expect(restored.conversation.id).toBe(before.conversation.id)
    expect(restored.entries).toEqual(before.entries)
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
  await expect(page.locator('#answer')).toContainText('1', { timeout: 120_000 })
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
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
