import type { Page } from '@playwright/test'
import { expect } from '../../fixtures/browser.js'
export { test, expect } from '../../fixtures/browser.js'

export async function load(page: Page, example: string, engine = 'worker', modelId?: string) {
  await page.getByLabel('Example', { exact: true }).selectOption(example)
  await page.getByLabel('Engine', { exact: true }).selectOption(engine)
  if (modelId) await page.getByLabel('Model', { exact: true }).selectOption(modelId)
  await page.getByRole('button', { name: 'Load model', exact: true }).click()
  await expect
    .poll(
      async () => {
        const status = await page.getByRole('status').textContent()
        console.log(status)
        return status
      },
      { timeout: 12 * 60 * 1000, intervals: [1000, 15000, 30000] },
    )
    .toBe('Model loaded')
}

export async function run(page: Page) {
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByRole('status')).not.toHaveText('Running…', { timeout: 120_000 })
  await expect(page.getByRole('status')).toHaveText('Done')
  await expect(page.getByLabel('Response', { exact: true })).not.toBeEmpty()
  console.log('Response:', await page.getByLabel('Response', { exact: true }).textContent())
}

export async function unload(page: Page) {
  await page.getByRole('button', { name: 'Unload', exact: true }).click()
  await expect(page.getByRole('status')).toHaveText('Model unloaded', { timeout: 30_000 })
}

export function silentWav() {
  // Half a second of mono PCM16 silence. This checks audio processing, not transcription quality.
  const samples = 8000
  const bytes = Buffer.alloc(44 + samples * 2)
  bytes.write('RIFF', 0)
  bytes.writeUInt32LE(bytes.length - 8, 4)
  bytes.write('WAVEfmt ', 8)
  bytes.writeUInt32LE(16, 16)
  bytes.writeUInt16LE(1, 20)
  bytes.writeUInt16LE(1, 22)
  bytes.writeUInt32LE(16000, 24)
  bytes.writeUInt32LE(32000, 28)
  bytes.writeUInt16LE(2, 32)
  bytes.writeUInt16LE(16, 34)
  bytes.write('data', 36)
  bytes.writeUInt32LE(samples * 2, 40)
  return { name: 'silence.wav', mimeType: 'audio/wav', buffer: bytes }
}
