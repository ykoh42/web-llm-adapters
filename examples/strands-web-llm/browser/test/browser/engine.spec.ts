import type { Page } from '@playwright/test'
import { test, expect } from '../../fixtures/page.js'
import type { models } from './checks/models.js'
import type { processor } from './checks/processor.js'
import type { state } from './checks/state.js'
import type { integrity } from './checks/integrity.js'

async function run(page: Page, name: string) {
  await page.goto('http://127.0.0.1:4173/test/browser/engine.html')
  await page.getByLabel('Check').selectOption(name)
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  await expect(page.getByRole('status')).toHaveText('Done', { timeout: 12 * 60 * 1000 })
  return (await page.getByLabel('Result').textContent())!
}

test('native state, progress, history, device and reset APIs work', async ({ page }) => {
  const result: Awaited<ReturnType<typeof state>> = JSON.parse(await run(page, 'state'))
  expect(result.callbackMatches).toBe(true)
  expect(result.progress).toContain(1)
  expect(result.text.toLowerCase()).toContain('pineapple')
  expect(result.message).toBe(result.rawText)
  expect(result.text).not.toContain('<think>')
  expect(result.text).not.toContain('</think>')
  expect(result.afterReset).toBe(result.message)
  expect(typeof result.gpu).toBe('string')
  expect(result.maxBuffer).toBeGreaterThan(0)
  expect(result.stats).toContain('tokens/sec')
  expect(result.messages).toBeGreaterThanOrEqual(4)
})

test('custom processors control main and worker token-level sampling', async ({ page }) => {
  const result: Awaited<ReturnType<typeof processor>> = JSON.parse(await run(page, 'processor'))
  expect(result.main.tokens).toEqual([42, 42, 42])
  expect(result.main.sampled).toEqual([42, 42, 42])
  expect(result.main.resetCalled).toBe(true)
  expect(result.main.cleared).toBe(true)
  expect(result.main.stats).toContain('tokens/sec')
  expect(result.worker.tokens).toEqual([42, 42, 42])
})

test('custom model IDs select among three simultaneously loaded models', async ({ page }) => {
  const result: Awaited<ReturnType<typeof models>> = JSON.parse(await run(page, 'models'))
  expect(result.selected).toEqual(['Qwen3-first', 'Qwen3-second'])
  expect(result.parallel.map((reply) => reply.model)).toEqual(['Qwen3-first', 'Qwen3-second'])
  expect(result.parallel.every((reply) => Boolean(reply.text))).toBe(true)
  expect(result.dimensions).toBe(384)
})

test('a custom manifest model verifies config, Wasm and tokenizer checksums', async ({ page }) => {
  const result: Awaited<ReturnType<typeof integrity>> = JSON.parse(await run(page, 'integrity'))
  expect(result.manifest).toBe('mlc-model-manifest.json')
  expect(result.text).toBeTruthy()
  expect(result.failures.map((failure) => failure.field)).toEqual([
    'config',
    'model_lib',
    'tokenizer',
  ])
  for (const failure of result.failures) {
    expect(failure.actual).not.toBe(failure.expected)
    expect(failure.url).toMatch(/^https:/)
  }
})

test('native generation resumes after an interruption', async ({ page }) => {
  const result: { text: string; reports: unknown[] } = JSON.parse(await run(page, 'resume'))
  expect(result.text).toContain('30')
  const details = JSON.stringify(result.reports, null, 2)
  expect(details).toContain('"recoveredText"')
  expect(details).toMatch(/"recoveryMode": "(kv|token_replay)"/)
})

test('native text completion returns two choices', async ({ page }) => {
  const result: { text: string; reports: unknown[] } = JSON.parse(await run(page, 'completions'))
  expect(result.text.split('\n\n---\n\n')).toHaveLength(2)
})

test('native embeddings return a finite vector', async ({ page }) => {
  const result: { text: string; reports: unknown[] } = JSON.parse(await run(page, 'embeddings'))
  const vector = JSON.parse(result.text) as {
    dimensions: number
    firstValues: number[]
  }
  expect(vector.dimensions).toBe(384)
  expect(vector.firstValues).toHaveLength(8)
  expect(vector.firstValues.every(Number.isFinite)).toBe(true)
})
