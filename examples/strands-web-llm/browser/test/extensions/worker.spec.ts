import { test, expect } from '../../fixtures/extension.js'
import { load, run, unload } from '../../fixtures/page.js'

test('Strands streams, stops and reuses a real Chrome extension worker', async ({
  page,
  extensionId,
}) => {
  await page.goto(`chrome-extension://${extensionId}/index.html`)
  await load(page, 'chat', 'extension')
  await page.getByLabel('Message', { exact: true }).fill('Count from 1 to 1000. Do not explain.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByLabel('Response', { exact: true })).not.toBeEmpty({ timeout: 120_000 })
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(page.getByRole('status')).toHaveText('Stopped', { timeout: 30_000 })
  await page.getByLabel('Message', { exact: true }).fill('Say hello in one short sentence.')
  await run(page)
  await unload(page)
})
